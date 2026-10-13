// =============================================================================
// Cardinal RHI — D3D12 async compute (AEGIS Block 10).
//
// The async-compute lane: a dedicated D3D12_COMMAND_LIST_TYPE_COMPUTE queue
// (created in device.cpp::create_queues) plus the three objects that drive it:
//
//   D3D12Fence         — ID3D12Fence + a Win32 event, the timeline sync object
//                        the host fences between the graphics + compute queues.
//   D3D12ComputeRecorder — a minimal Swapchain that records the compute-queue-
//                        valid ops (buffer copies, UAV barriers, debug markers)
//                        onto the async command list. Present/frame/viewport
//                        entry points are stubbed (never reached during a
//                        ComputeQueue::submit recording).
//   D3D12ComputeQueue  — owns a COMPUTE allocator + list; submit() resets the
//                        list, runs the caller's RecordFn against the recorder,
//                        closes + ExecuteCommandLists on the compute queue, then
//                        signals the caller's Fence. An internal fence makes the
//                        allocator safe to reuse across back-to-back submits.
//
// The recorder is dispatch-complete: bind_pipeline (compute root sig + PSO),
// root SRV/UAV buffer binds, inline push constants (<=64B blocks), dispatch,
// copies, UAV barriers and the cross-queue timeline handshake — enough to run
// any AEGIS kernel off the graphics queue (and for the headless GPU
// validation harness in tests/gpu_compute).
//
// Windows-only.
// =============================================================================

#if defined(_WIN32) && !defined(CARDINAL_NO_D3D12)

#include "internal.hpp"
#include <cardinal/core/clock/stopwatch.hpp>

#if CARDINAL_PLATFORM_WINDOWS

namespace cardinal::rhi {

namespace {

// PIX / RenderDoc string-encoding selector (1 = ANSI), matching commands.cpp.
constexpr UINT kPixAnsi = 1u;

// ---------------------------------------------------------------------------
// D3D12ComputeRecorder — records compute-queue-valid ops onto `cmd_`.
// ---------------------------------------------------------------------------
class D3D12ComputeRecorder final : public Swapchain {
public:
    D3D12ComputeRecorder(ID3D12Device* dev, ID3D12GraphicsCommandList* cmd,
                         ID3D12QueryHeap* timestamps, u32 heap_base, u32 timestamp_cap)
        : dev_(dev), cmd_(cmd), qh_(timestamps), qh_base_(heap_base), qh_cap_(timestamp_cap) {}

    // Per-pass GPU timing into the lane's timestamp heap (this slot's range).
    // TIMESTAMP is the one query type a COMPUTE list may issue and resolve.
    void write_timestamp(u32 /*heap*/, u32 index) override {
        if (!cmd_ || !qh_ || index >= qh_cap_) return;
        cmd_->EndQuery(qh_, D3D12_QUERY_TYPE_TIMESTAMP, qh_base_ + index);
        if (index + 1 > qh_used_) qh_used_ = index + 1;
    }
    u32 timestamps_written() const noexcept { return qh_used_; }

    // ---- real ops (valid on a COMPUTE command list) ----
    void copy_buffer(Buffer* src, usize src_off,
                     Buffer* dst, usize dst_off, usize size) override {
        auto* s = static_cast<D3D12Buffer*>(src);
        auto* d = static_cast<D3D12Buffer*>(dst);
        if (!cmd_ || !s || !d) return;
        cmd_->CopyBufferRegion(d->resource(), dst_off, s->resource(), src_off, size);
    }
    void uav_barrier(Buffer* b) override {
        auto* db = static_cast<D3D12Buffer*>(b);
        if (!cmd_ || !db) return;
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        bar.UAV.pResource = db->resource();
        cmd_->ResourceBarrier(1, &bar);
    }
    void transition_buffer_state(Buffer* b, ResourceState before,
                                 ResourceState after) override {
        auto* db = static_cast<D3D12Buffer*>(b);
        if (!cmd_ || !db || before == after) return;
        // Compute-list-legal subset of the main swapchain's state map.
        auto st = [](ResourceState s) -> D3D12_RESOURCE_STATES {
            switch (s) {
                case ResourceState::UnorderedAccess:  return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                case ResourceState::ShaderResource:   return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                case ResourceState::CopySource:       return D3D12_RESOURCE_STATE_COPY_SOURCE;
                case ResourceState::CopyDest:         return D3D12_RESOURCE_STATE_COPY_DEST;
                default:                              return D3D12_RESOURCE_STATE_COMMON;
            }
        };
        D3D12_RESOURCE_BARRIER bar{};
        bar.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource   = db->resource();
        bar.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        bar.Transition.StateBefore = st(before);
        bar.Transition.StateAfter  = st(after);
        cmd_->ResourceBarrier(1, &bar);
    }
    void begin_event(const char* n) override {
        if (cmd_ && n) cmd_->BeginEvent(kPixAnsi, n,
                                        static_cast<UINT>(cardinal::strlen(n) + 1u));
    }
    void end_event() override { if (cmd_) cmd_->EndEvent(); }
    void insert_marker(const char* n) override {
        if (cmd_ && n) cmd_->SetMarker(kPixAnsi, n,
                                       static_cast<UINT>(cardinal::strlen(n) + 1u));
    }

    // ---- present / frame / viewport surface — never reached here ----
    u32    width()  const noexcept override { return 0; }
    u32    height() const noexcept override { return 0; }
    Format color_format() const noexcept override { return Format::B8G8R8A8_UNORM; }
    void   set_vsync(bool) override {}
    bool   vsync() const noexcept override { return false; }
    void   set_vsync_interval(u32) override {}
    u32    vsync_interval() const noexcept override { return 0; }
    bool   resize(u32, u32) override { return false; }
    void   set_on_rebuilt(OnRebuilt) override {}
    void   set_viewport_size(u32, u32) override {}
    u32    viewport_width()  const noexcept override { return 0; }
    u32    viewport_height() const noexcept override { return 0; }
    void   set_viewport_size(u32, u32, u32) override {}
    u32    viewport_width (u32) const noexcept override { return 0; }
    u32    viewport_height(u32) const noexcept override { return 0; }
    void   set_viewport_count(u32) override {}
    u32    viewport_count() const noexcept override { return 0; }
    void   set_active_viewport(u32) override {}
    u32    active_viewport() const noexcept override { return 0; }
    u32    begin_frame(float, float, float, float) override { return 0; }
    void   end_frame() override {}
    void   set_overlay(OverlayCallback, void*) override {}
    // Real compute recording on the async list.
    void   bind_pipeline(Pipeline* p) override {
        auto* dp = static_cast<D3D12Pipeline*>(p);
        if (!cmd_ || dp == nullptr || !dp->is_compute()) return;
        bound_ = dp;
        cmd_->SetComputeRootSignature(dp->root_signature());
        cmd_->SetPipelineState(dp->pso());
    }
    void   dispatch(u32 gx, u32 gy = 1, u32 gz = 1) override {
        if (cmd_) cmd_->Dispatch(gx, gy, gz);
    }
    void   bind_storage_buffer(u32 slot, Buffer* b) override {
        auto* dp = static_cast<D3D12Pipeline*>(bound_); auto* db = static_cast<D3D12Buffer*>(b);
        if (!cmd_ || !dp || !db || slot >= dp->storage_buffer_slots()) return;
        cmd_->SetComputeRootShaderResourceView(dp->storage_root_base() + slot,
                                               db->resource()->GetGPUVirtualAddress());
    }
    void   bind_storage_buffer_uav(u32 slot, Buffer* b) override {
        auto* dp = static_cast<D3D12Pipeline*>(bound_); auto* db = static_cast<D3D12Buffer*>(b);
        if (!cmd_ || !dp || !db || slot >= dp->uav_slots()) return;
        cmd_->SetComputeRootUnorderedAccessView(dp->uav_root_base() + slot,
                                                db->resource()->GetGPUVirtualAddress());
    }
    void   bind_vertex_buffer(Buffer*, usize) override {}
    void   draw(u32, u32, u32, u32) override {}
    void   set_push_constants(u32 offset, const void* data, u32 size) override {
        // Inline-root-constants path only (blocks <=64B — every AEGIS kernel).
        // The CBV-ring path for large blocks lives on the main swapchain; the
        // async recorder has no per-frame ring, so big blocks warn + drop.
        if (!cmd_ || bound_ == nullptr) return;
        const u32 declared = bound_->push_constant_size();
        if (declared == 0) return;
        if (offset + size > declared || bound_->push_is_cbv()) {
            static bool warned = false;
            if (!warned) {
                cardinal::log::warnf("rhi/d3d12",
                    "async set_push_constants unsupported shape "
                    "(offset=%u size=%u declared=%u cbv=%d) — dropped",
                    offset, size, declared, bound_->push_is_cbv() ? 1 : 0);
                warned = true;
            }
            return;
        }
        cmd_->SetComputeRoot32BitConstants(0, (size + 3u) / 4u, data, offset / 4u);
    }

private:
    ID3D12Device*              dev_{nullptr};
    ID3D12GraphicsCommandList* cmd_{nullptr};
    ID3D12QueryHeap*           qh_{nullptr};      // lane timestamp heap
    u32                        qh_base_{0};       // this slot's first query
    u32                        qh_cap_{0};
    u32                        qh_used_{0};       // highest index written + 1
    D3D12Pipeline*             bound_{nullptr};   // for SetComputeRoot* binding
};

// ---------------------------------------------------------------------------
// D3D12Fence — ID3D12Fence timeline + a Win32 event for CPU waits.
// ---------------------------------------------------------------------------
class D3D12Fence final : public Fence {
public:
    bool initialize(ID3D12Device* dev, u64 initial) {
        if (!dev) return false;
        if (FAILED(dev->CreateFence(initial, D3D12_FENCE_FLAG_NONE,
                                    IID_PPV_ARGS(&fence_)))) return false;
        event_ = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        next_  = initial + 1u;
        return event_ != nullptr;
    }
    ~D3D12Fence() override { if (event_) CloseHandle(event_); }

    u64 wait_cpu(u64 value, u64 timeout_ns) override {
        if (!fence_) return 0;
        if (fence_->GetCompletedValue() < value) {
            if (FAILED(fence_->SetEventOnCompletion(value, event_)))
                return fence_->GetCompletedValue();
            const DWORD ms = (timeout_ns == 0) ? INFINITE
                           : static_cast<DWORD>(timeout_ns / 1'000'000ull);
            WaitForSingleObject(event_, ms);
        }
        return fence_->GetCompletedValue();
    }
    u64 current_value() const noexcept override {
        return fence_ ? fence_->GetCompletedValue() : 0u;
    }

    ID3D12Fence* d3d()  const noexcept { return fence_.Get(); }
    u64          bump()       noexcept { return next_++; }   // auto-increment for signal_fence

private:
    ComPtr<ID3D12Fence> fence_;
    HANDLE              event_{nullptr};
    u64                 next_{1};
};

// ---------------------------------------------------------------------------
// D3D12ComputeQueue — records + submits onto the device's dedicated COMPUTE
// queue through a ring of kRing command allocators: submit() waits only for
// the allocator's previous submission (N - kRing), so the host records frame
// N while N-1 still executes. One command list suffices (a list may be reset
// right after ExecuteCommandLists; only its allocator must wait for the GPU).
// Lists on one queue execute in order and ExecuteCommandLists boundaries make
// prior UAV writes visible (that is what lets resources decay to COMMON
// there), so no cross-list barrier is needed — unlike the Vulkan lane.
// ---------------------------------------------------------------------------
class D3D12ComputeQueue final : public ComputeQueue {
public:
    static constexpr u32 kRing         = 2;
    static constexpr u32 kTimestampCap = 128;   // 64 timed passes per submission

    bool initialize(ID3D12Device* dev, ID3D12CommandQueue* queue) {
        if (!dev || !queue) return false;
        dev_ = dev; queue_ = queue;
        // Per-pass timestamps: one heap with a kTimestampCap range per ring
        // slot, resolved into a READBACK buffer at the end of each list and
        // read once that submission's fence retired. Frequency is the
        // COMPUTE queue's (it can differ from the graphics queue's).
        if (SUCCEEDED(queue->GetTimestampFrequency(&ts_freq_)) && ts_freq_ > 0) {
            // GPU<->CPU clock pair for submission-start latency (see
            // last_retired_start_latency_ns); failure just leaves it at 0.
            LARGE_INTEGER qpf{};
            QueryPerformanceFrequency(&qpf);
            qpc_freq_ = static_cast<u64>(qpf.QuadPart);
            if (FAILED(queue->GetClockCalibration(&cal_gpu_, &cal_cpu_))) { cal_gpu_ = 0; cal_cpu_ = 0; }
            D3D12_QUERY_HEAP_DESC qd{};
            qd.Type  = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
            qd.Count = kRing * kTimestampCap;
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC rd{};
            rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            rd.Width            = u64(kRing) * kTimestampCap * sizeof(u64);
            rd.Height           = 1;
            rd.DepthOrArraySize = 1;
            rd.MipLevels        = 1;
            rd.SampleDesc.Count = 1;
            rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            if (FAILED(dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&ts_heap_))) ||
                FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&ts_readback_)))) {
                ts_heap_.Reset(); ts_readback_.Reset(); ts_freq_ = 0;   // timing off, lane works
            }
        } else {
            ts_freq_ = 0;
        }
        for (u32 i = 0; i < kRing; ++i)
            if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                   IID_PPV_ARGS(&alloc_[i])))) return false;
        if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                          alloc_[0].Get(), nullptr,
                                          IID_PPV_ARGS(&list_)))) return false;
        list_->Close();   // submit() resets before recording
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                    IID_PPV_ARGS(&done_)))) return false;
        done_event_ = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        return done_event_ != nullptr;
    }
    ~D3D12ComputeQueue() override {
        // An allocator may not be released while the GPU still reads it.
        if (done_ && done_event_ && last_signal_ != 0 &&
            done_->GetCompletedValue() < last_signal_) {
            if (SUCCEEDED(done_->SetEventOnCompletion(last_signal_, done_event_)))
                WaitForSingleObject(done_event_, INFINITE);
        }
        if (done_event_) CloseHandle(done_event_);
    }

    u32 frames_in_flight() const noexcept override { return kRing; }

    u64 last_submit_wait_ns() const noexcept override { return last_wait_ns_; }
    u64 last_retired_start_latency_ns() const noexcept override { return retired_latency_ns_; }
    u32 timestamp_capacity() const noexcept override { return ts_heap_ ? kTimestampCap : 0u; }
    u64 timestamp_ticks_per_second() const noexcept override { return ts_freq_; }
    bool retired_timestamps(u64& signal_value, cardinal::vector<u64>& ticks) const override {
        if (retired_value_ == 0) return false;
        signal_value = retired_value_;
        ticks        = retired_ticks_;
        return true;
    }

    u64 submit(RecordFn record, void* user,
               Fence* signal_fence, u64 signal_value) override {
        if (!queue_ || !list_ || !alloc_[0]) return 0;
        const u32 slot = cur_;
        cur_ = (cur_ + 1) % kRing;

        // Wait for THIS allocator's previous submission (N - kRing) to retire
        // before resetting it; N-1 keeps executing.
        last_wait_ns_ = 0;
        if (slot_signal_[slot] != 0 && done_->GetCompletedValue() < slot_signal_[slot]) {
            const cardinal::core::Stopwatch wait_sw;
            if (SUCCEEDED(done_->SetEventOnCompletion(slot_signal_[slot], done_event_)))
                WaitForSingleObject(done_event_, INFINITE);
            last_wait_ns_ = wait_sw.elapsed_ns();
        }
        // That submission's resolved timestamps are final: read its range.
        if (ts_readback_ && slot_ts_written_[slot] > 0) {
            const usize first = usize(slot) * kTimestampCap;
            D3D12_RANGE rr{ first * sizeof(u64), (first + slot_ts_written_[slot]) * sizeof(u64) };
            void* p = nullptr;
            if (SUCCEEDED(ts_readback_->Map(0, &rr, &p)) && p) {
                const u64* src = static_cast<const u64*>(p) + first;
                retired_ticks_.assign(src, src + slot_ts_written_[slot]);
                retired_latency_ns_ = 0;
                if (cal_cpu_ != 0 && qpc_freq_ != 0 && slot_submit_qpc_[slot] != 0 && !retired_ticks_.empty()) {
                    const double gpu_ns = (static_cast<double>(retired_ticks_[0]) - static_cast<double>(cal_gpu_))
                                        * 1.0e9 / static_cast<double>(ts_freq_);
                    const double cpu_ns = (static_cast<double>(slot_submit_qpc_[slot]) - static_cast<double>(cal_cpu_))
                                        * 1.0e9 / static_cast<double>(qpc_freq_);
                    const double lat = gpu_ns - cpu_ns;
                    retired_latency_ns_ = lat > 0.0 ? static_cast<u64>(lat) : 0u;
                }
                D3D12_RANGE none{ 0, 0 };
                ts_readback_->Unmap(0, &none);
                retired_value_ = slot_sig_value_[slot];
            }
        }

        if (FAILED(alloc_[slot]->Reset())) return 0;
        if (FAILED(list_->Reset(alloc_[slot].Get(), nullptr))) return 0;
        u32 ts_written = 0;
        if (record) {
            D3D12ComputeRecorder rec(dev_, list_.Get(), ts_heap_.Get(),
                                     slot * kTimestampCap, ts_heap_ ? kTimestampCap : 0u);
            record(&rec, user);
            ts_written = rec.timestamps_written();
        }
        if (ts_written > 0 && ts_heap_ && ts_readback_)
            list_->ResolveQueryData(ts_heap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
                                    slot * kTimestampCap, ts_written, ts_readback_.Get(),
                                    u64(slot) * kTimestampCap * sizeof(u64));
        slot_ts_written_[slot] = ts_written;
        slot_sig_value_[slot]  = signal_value;
        if (FAILED(list_->Close())) return 0;

        ID3D12CommandList* lists[] = { list_.Get() };
        queue_->ExecuteCommandLists(1, lists);
        {
            LARGE_INTEGER now{};
            QueryPerformanceCounter(&now);
            slot_submit_qpc_[slot] = static_cast<u64>(now.QuadPart);
        }

        // Caller's cross-queue fence first, then our internal reuse fence.
        if (auto* df = static_cast<D3D12Fence*>(signal_fence); df && df->d3d())
            queue_->Signal(df->d3d(), signal_value);
        queue_->Signal(done_.Get(), ++last_signal_);
        slot_signal_[slot] = last_signal_;
        return signal_value;
    }

private:
    ID3D12Device*                     dev_{nullptr};
    ID3D12CommandQueue*               queue_{nullptr};   // device-owned COMPUTE queue
    ComPtr<ID3D12CommandAllocator>    alloc_[kRing];
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence>               done_;             // internal allocator-reuse fence
    HANDLE                            done_event_{nullptr};
    u64                               last_signal_{0};
    u64                               slot_signal_[kRing]{};
    u32                               cur_{0};
    // Per-pass timestamps (see initialize).
    ComPtr<ID3D12QueryHeap>           ts_heap_;
    ComPtr<ID3D12Resource>            ts_readback_;
    u64                               ts_freq_{0};
    u64                               cal_gpu_{0}, cal_cpu_{0}, qpc_freq_{0};   // GetClockCalibration pair
    u64                               slot_submit_qpc_[kRing]{};
    u64                               retired_latency_ns_{0};
    u32                               slot_ts_written_[kRing]{};
    u64                               slot_sig_value_[kRing]{};
    u64                               retired_value_{0};
    cardinal::vector<u64>             retired_ticks_;
    u64                               last_wait_ns_{0};
};

}  // namespace

// ---------------------------------------------------------------------------
// Device factory overrides
// ---------------------------------------------------------------------------
cardinal::unique_ptr<Fence> D3D12Device::create_fence(u64 initial_value) {
    auto f = cardinal::make_unique<D3D12Fence>();
    if (!f->initialize(device_.Get(), initial_value)) return nullptr;
    return f;
}

cardinal::unique_ptr<ComputeQueue> D3D12Device::create_async_compute_queue() {
    if (!compute_queue_) return nullptr;          // caps.async_compute == false
    auto q = cardinal::make_unique<D3D12ComputeQueue>();
    if (!q->initialize(device_.Get(), compute_queue_.Get())) return nullptr;
    return q;
}

// ---------------------------------------------------------------------------
// Swapchain (graphics-queue) side of the cross-queue handshake
// ---------------------------------------------------------------------------
u64 D3D12Swapchain::signal_fence(Fence* f) {
    auto* df = static_cast<D3D12Fence*>(f);
    if (!df || !df->d3d() || !dev_.gfx_queue()) return 0;
    const u64 v = df->bump();
    dev_.gfx_queue()->Signal(df->d3d(), v);
    return v;
}

void D3D12Swapchain::wait_fence(Fence* f, u64 value) {
    auto* df = static_cast<D3D12Fence*>(f);
    if (!df || !df->d3d() || !dev_.gfx_queue()) return;
    dev_.gfx_queue()->Wait(df->d3d(), value);     // GPU-side queue wait
}

}  // namespace cardinal::rhi

#endif  // CARDINAL_PLATFORM_WINDOWS
#endif  // _WIN32 && !CARDINAL_NO_D3D12
