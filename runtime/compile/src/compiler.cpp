// =============================================================================
// Cardinal — async compile worker.
//
// Single background thread + a bounded job queue. Each job is run inside a
// __try / __except barrier on Windows (SEH) so an access-violation inside
// DXC turns into a CompileResult{.ok=false, .crashed=true} rather than
// taking the editor down. POSIX builds use sigaction-based SIGSEGV
// handling on a per-thread basis (compiled out for now — Phase 4-C2).
// =============================================================================

#include <cardinal/compile/compiler.hpp>
#include <cardinal/core/diag/log.hpp>
#include <cardinal/core/platform.hpp>

#include <cardinal/core/std/chrono.hpp>
#include <cardinal/core/std/containers.hpp>
#include <cardinal/core/std/cstdio.hpp>
#include <cardinal/core/std/future.hpp>
#include <cardinal/core/std/thread.hpp>
#include <cardinal/core/std/utility.hpp>

#if CARDINAL_PLATFORM_WINDOWS
#include <Windows.h>
#include <unknwn.h>
#include <dxcapi.h>
#include <wrl/client.h>
#pragma comment(lib, "dxcompiler.lib")
using Microsoft::WRL::ComPtr;
#endif

namespace cardinal::compile {

CompileFuture::CompileFuture(cardinal::future<CompileResult> fut) : fut_(cardinal::move(fut)) {}
bool CompileFuture::is_ready() const noexcept {
    return fut_.valid() &&
           fut_.wait_for(cardinal::chrono::seconds(0)) == cardinal::future_status::ready;
}
CompileResult CompileFuture::take() {
    if (!fut_.valid()) return CompileResult{ false, {}, "future already consumed", false, 0 };
    return fut_.get();
}

namespace {

#if CARDINAL_PLATFORM_WINDOWS

// ---- DXC singletons (per worker) -------------------------------------------
struct Dxc {
    ComPtr<IDxcUtils>          utils;
    ComPtr<IDxcCompiler3>      compiler;
    ComPtr<IDxcIncludeHandler> includes;
    bool ready{false};

    bool init() {
        if (FAILED(DxcCreateInstance(CLSID_DxcUtils,    IID_PPV_ARGS(&utils))))    return false;
        if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)))) return false;
        if (FAILED(utils->CreateDefaultIncludeHandler(&includes)))                  return false;
        ready = true;
        return true;
    }
};

// Compile a single request. Returns false on DXC-reported failure (errors
// is then populated). Crashes are handled by the SEH wrapper around this.
bool compile_one(Dxc& d, const CompileRequest& req,
                 cardinal::vector<u8>& out_bytes, cardinal::string& out_err)
{
    LPCWSTR target = L"vs_6_5";
    switch (req.stage) {
        case rhi::ShaderStage::Vertex:   target = L"vs_6_5"; break;
        case rhi::ShaderStage::Fragment: target = L"ps_6_5"; break;
        case rhi::ShaderStage::Compute:  target = L"cs_6_5"; break;
    }
    wchar_t entry_w[64]{};
    MultiByteToWideChar(CP_UTF8, 0, req.entry_point.c_str(), -1, entry_w, 63);

    DxcBuffer src{};
    src.Ptr      = req.source.data();
    src.Size     = req.source.size();
    src.Encoding = DXC_CP_UTF8;

    LPCWSTR args[] = {
        L"-E", entry_w,
        L"-T", target,
        L"-Zi",
        L"-Qstrip_reflect",
        L"-HV", L"2021",
    };

    ComPtr<IDxcResult> result;
    HRESULT hr = d.compiler->Compile(&src, args, _countof(args),
                                     d.includes.Get(), IID_PPV_ARGS(&result));
    if (FAILED(hr)) { out_err = "DXC Compile call returned failure HRESULT"; return false; }
    HRESULT status = S_OK; result->GetStatus(&status);
    if (FAILED(status)) {
        ComPtr<IDxcBlobUtf8> errs;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errs), nullptr);
        out_err = (errs && errs->GetStringLength() > 0)
                  ? cardinal::string(errs->GetStringPointer(), errs->GetStringLength())
                  : "DXC failed without error blob";
        return false;
    }
    ComPtr<IDxcBlob> code;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&code), nullptr);
    if (!code || code->GetBufferSize() == 0) { out_err = "DXC empty object"; return false; }
    out_bytes.assign(static_cast<const u8*>(code->GetBufferPointer()),
                     static_cast<const u8*>(code->GetBufferPointer()) + code->GetBufferSize());
    return true;
}

// MSVC: __try / __except cannot coexist with C++ destructors in the same
// function. Split the protected call into a no-destructors helper that the
// outer (RAII-using) function calls through.
struct SehArgs {
    Dxc*                     dxc;
    const CompileRequest*    req;
    cardinal::vector<u8>*         out_bytes;
    cardinal::string*             out_err;
};

// All RAII has been hoisted to the caller; this function holds only PODs.
DWORD invoke_compile_seh(SehArgs* a, bool* out_ok) noexcept {
    DWORD code = 0;
    __try {
        *out_ok = compile_one(*a->dxc, *a->req, *a->out_bytes, *a->out_err);
        return 0;
    } __except (code = GetExceptionCode(), EXCEPTION_EXECUTE_HANDLER) {
        return code;
    }
}

CompileResult run_with_seh(Dxc& d, const CompileRequest& req) {
    CompileResult r{};
    cardinal::vector<u8> bytes;
    cardinal::string err;
    SehArgs args{ &d, &req, &bytes, &err };
    bool ok = false;
    const auto t0 = cardinal::chrono::steady_clock::now();
    const DWORD seh = invoke_compile_seh(&args, &ok);
    r.elapsed_us = static_cast<u64>(cardinal::chrono::duration_cast<cardinal::chrono::microseconds>(
                       cardinal::chrono::steady_clock::now() - t0).count());
    if (seh != 0) {
        char buf[128];
        cardinal::snprintf(buf, sizeof(buf),
            "compiler crashed (SEH 0x%08lx) on %s",
            seh, req.filename.empty() ? req.entry_point.c_str() : req.filename.c_str());
        r.ok       = false;
        r.crashed  = true;
        r.error    = buf;
    } else {
        r.ok        = ok;
        r.blob.bytes= cardinal::move(bytes);
        r.error     = cardinal::move(err);
    }
    return r;
}

#else  // POSIX — Phase 4-C2 will add sigaction-based isolation.
struct Dxc { bool ready{false}; bool init() { return false; } };
CompileResult run_with_seh(Dxc&, const CompileRequest& req) {
    return { false, {}, "compile worker only implemented on Windows for now", false, 0 };
}
#endif  // CARDINAL_PLATFORM_WINDOWS

// ---- Worker implementation -------------------------------------------------
class WorkerImpl final : public Worker {
public:
    WorkerImpl() {
        if (!dxc_.init()) {
            cardinal::log::warnf("compile",
                "DXC initialisation failed — compile worker is degraded");
        }
        thread_ = cardinal::thread([this]{ run(); });
    }
    ~WorkerImpl() override { shutdown(); }

    CompileFuture submit(const CompileRequest& req) override {
        cardinal::promise<CompileResult> prom;
        auto fut = prom.get_future();
        {
            cardinal::lock_guard lk(mu_);
            if (stopping_) {
                CompileResult err{ false, {}, "worker is shutting down", false, 0 };
                prom.set_value(cardinal::move(err));
                return CompileFuture(cardinal::move(fut));
            }
            queue_.push_back({ req, cardinal::move(prom) });
        }
        cv_.notify_one();
        return CompileFuture(cardinal::move(fut));
    }

    u32  pending() const noexcept override {
        cardinal::lock_guard lk(mu_);
        return static_cast<u32>(queue_.size());
    }

    void shutdown() override {
        {
            cardinal::lock_guard lk(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    struct Job {
        CompileRequest             req;
        cardinal::promise<CompileResult> prom;
    };

    void run() {
        for (;;) {
            Job j;
            {
                cardinal::unique_lock lk(mu_);
                cv_.wait(lk, [&]{ return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                j = cardinal::move(queue_.front());
                queue_.pop_front();
            }
            CompileResult r = run_with_seh(dxc_, j.req);
            if (r.crashed) {
                cardinal::log::errorf("compile",
                    "worker survived crash: %s", r.error.c_str());
            }
            j.prom.set_value(cardinal::move(r));
        }
    }

    Dxc                                 dxc_;
    cardinal::thread                         thread_;
    mutable cardinal::mutex                  mu_;
    cardinal::condition_variable             cv_;
    cardinal::deque<Job>                     queue_;
    bool                                stopping_{false};
};

}  // namespace

cardinal::unique_ptr<Worker> Worker::create() {
    return cardinal::make_unique<WorkerImpl>();
}

}  // namespace cardinal::compile
