// =============================================================================
// Cardinal — Audio output backend.
//
// Windows: WASAPI shared-mode, event-driven render thread. The thread
// owns the COM device objects, waits on the render event, and fills
// each buffer by calling Engine::render() — the engine never sees an
// OS audio API. Other platforms: a no-op stub (ALSA/CoreAudio later),
// so callers get silence rather than a link error.
//
// Single TU, both paths behind one CARDINAL_PLATFORM_WINDOWS switch —
// mirrors the input_windows.cpp split. CMake links ole32 on Windows.
// =============================================================================

#include <cardinal/audio/audio.hpp>
#include <cardinal/core/log.hpp>

#if CARDINAL_PLATFORM_WINDOWS

#include <atomic>
#include <thread>

#ifndef WIN32_LEAN_AND_MEAN          // already defined project-wide
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <mmreg.h>          // WAVE_FORMAT_IEEE_FLOAT / WAVEFORMATEX
#include <mmdeviceapi.h>    // IMMDeviceEnumerator / eRender
#include <audioclient.h>    // IAudioClient / IAudioRenderClient
#include <ksmedia.h>        // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT

namespace cardinal::audio {

namespace {

// Release a COM pointer + null it.
template <class T> void safe_release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

class WasapiOutput final : public OutputBackend {
public:
    explicit WasapiOutput(std::shared_ptr<Engine> engine)
        : engine_(std::move(engine)) {}

    ~WasapiOutput() override {
        stop_.store(true, std::memory_order_release);
        if (event_) SetEvent(event_);          // wake the render wait
        if (thread_.joinable()) thread_.join();
        // Device objects are created + destroyed on the audio thread;
        // by here it has joined so they're already released.
        if (event_) { CloseHandle(event_); event_ = nullptr; }
    }

    bool start() {
        thread_ = std::thread([this]{ run_(); });
        return true;   // failures are logged on the thread; degrade to silence
    }

private:
    void run_() {
        // The audio thread owns its own COM apartment (MTA — WASAPI is
        // fine multithreaded and we don't want to depend on the caller's
        // apartment model).
        const HRESULT co = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        const bool co_ok = SUCCEEDED(co);

        IMMDeviceEnumerator* enumr   = nullptr;
        IMMDevice*           device  = nullptr;
        IAudioClient*        client  = nullptr;
        IAudioRenderClient*  render  = nullptr;
        WAVEFORMATEX*        mix_fmt = nullptr;

        auto cleanup = [&]{
            if (client) client->Stop();
            safe_release(render);
            safe_release(client);
            safe_release(device);
            safe_release(enumr);
            if (mix_fmt) { CoTaskMemFree(mix_fmt); mix_fmt = nullptr; }
            if (co_ok) CoUninitialize();
        };

        if (FAILED(CoCreateInstance(
                __uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                __uuidof(IMMDeviceEnumerator),
                reinterpret_cast<void**>(&enumr)))) {
            cardinal::log::warnf("audio/wasapi",
                "MMDeviceEnumerator unavailable — audio silent");
            cleanup(); return;
        }
        if (FAILED(enumr->GetDefaultAudioEndpoint(
                eRender, eConsole, &device))) {
            cardinal::log::warnf("audio/wasapi",
                "no default render endpoint — audio silent");
            cleanup(); return;
        }
        if (FAILED(device->Activate(
                __uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                reinterpret_cast<void**>(&client)))) {
            cardinal::log::warnf("audio/wasapi",
                "IAudioClient activate failed — audio silent");
            cleanup(); return;
        }
        if (FAILED(client->GetMixFormat(&mix_fmt)) || mix_fmt == nullptr) {
            cardinal::log::warnf("audio/wasapi",
                "GetMixFormat failed — audio silent");
            cleanup(); return;
        }

        // We only feed interleaved 32-bit float (what Engine::render
        // produces). Shared-mode mixes are almost always float32; if
        // this device isn't, bail rather than emit noise.
        const bool is_float =
            (mix_fmt->wFormatTag == WAVE_FORMAT_IEEE_FLOAT) ||
            (mix_fmt->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
             reinterpret_cast<WAVEFORMATEXTENSIBLE*>(mix_fmt)
                 ->SubFormat == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT);
        if (!is_float || mix_fmt->wBitsPerSample != 32) {
            cardinal::log::warnf("audio/wasapi",
                "device mix format not 32-bit float — audio silent");
            cleanup(); return;
        }
        const u32 channels = mix_fmt->nChannels;

        // 30 ms shared buffer; event-driven so the thread sleeps until
        // the mixer needs more data.
        const REFERENCE_TIME buf_dur = 300000;   // 100-ns units = 30 ms
        if (FAILED(client->Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                buf_dur, 0, mix_fmt, nullptr))) {
            cardinal::log::warnf("audio/wasapi",
                "IAudioClient::Initialize failed — audio silent");
            cleanup(); return;
        }
        event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (event_ == nullptr ||
            FAILED(client->SetEventHandle(event_))) {
            cardinal::log::warnf("audio/wasapi",
                "SetEventHandle failed — audio silent");
            cleanup(); return;
        }
        UINT32 buffer_frames = 0;
        if (FAILED(client->GetBufferSize(&buffer_frames)) ||
            FAILED(client->GetService(
                __uuidof(IAudioRenderClient),
                reinterpret_cast<void**>(&render)))) {
            cardinal::log::warnf("audio/wasapi",
                "render service unavailable — audio silent");
            cleanup(); return;
        }
        if (FAILED(client->Start())) {
            cardinal::log::warnf("audio/wasapi",
                "IAudioClient::Start failed — audio silent");
            cleanup(); return;
        }

        cardinal::log::infof("audio/wasapi",
            "output started: %u ch, %u Hz, %u-frame buffer",
            channels, mix_fmt->nSamplesPerSec, buffer_frames);

        // Render loop — wake on the WASAPI event, fill the unpadded
        // remainder of the shared buffer from the engine.
        while (!stop_.load(std::memory_order_acquire)) {
            if (WaitForSingleObject(event_, 200) != WAIT_OBJECT_0) {
                continue;   // timeout: re-check stop_, keep the device alive
            }
            if (stop_.load(std::memory_order_acquire)) break;
            UINT32 padding = 0;
            if (FAILED(client->GetCurrentPadding(&padding))) continue;
            const UINT32 avail = buffer_frames - padding;
            if (avail == 0) continue;
            BYTE* data = nullptr;
            if (FAILED(render->GetBuffer(avail, &data)) || data == nullptr) {
                continue;
            }
            engine_->render(reinterpret_cast<float*>(data),
                            avail, channels);
            render->ReleaseBuffer(avail, 0);
        }
        cleanup();
    }

    std::shared_ptr<Engine> engine_;
    std::thread             thread_;
    std::atomic<bool>       stop_{false};
    HANDLE                  event_{nullptr};
};

}  // namespace

std::unique_ptr<OutputBackend>
start_default_output(std::shared_ptr<Engine> engine) {
    if (!engine) return nullptr;
    auto out = std::make_unique<WasapiOutput>(std::move(engine));
    if (!out->start()) return nullptr;
    return out;
}

}  // namespace cardinal::audio

#else   // !CARDINAL_PLATFORM_WINDOWS

namespace cardinal::audio {

std::unique_ptr<OutputBackend>
start_default_output(std::shared_ptr<Engine> /*engine*/) {
    cardinal::log::infof("audio",
        "no output backend on this platform yet — audio silent");
    return nullptr;
}

}  // namespace cardinal::audio

#endif  // CARDINAL_PLATFORM_WINDOWS
