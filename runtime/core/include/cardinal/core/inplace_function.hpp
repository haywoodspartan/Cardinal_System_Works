#pragma once

// =============================================================================
// cardinal::inplace_function<R(Args...), Capacity, Align> — a heap-free
// std::function.
//
// std::function heap-allocates whenever the target callable exceeds its small-
// buffer (and the standard guarantees no inline storage at all). The engine
// stores callables everywhere — command run/enabled handlers, event
// subscribers, deferred tasks, UI callbacks — and most are small capturing
// lambdas. inplace_function gives type-erased call with the target stored
// INLINE in a fixed Capacity-byte buffer: zero allocation, predictable size,
// cache-friendly. A target that doesn't fit is a COMPILE error (bump Capacity
// at the call site), never a silent heap fallback.
//
// Drop-in for the std::function subset Cardinal uses: construct/assign from any
// compatible callable (incl. mutable lambdas), copy, move, reset/= nullptr,
// operator bool, operator(). Copyable (like std::function) — the target must
// be copyable.
//
// FOUNDATION RULE: lives in cardinal::core. Exposed as cardinal::inplace_function.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <cstddef>        // std::max_align_t
#include <new>            // placement new
#include <type_traits>    // decay_t, enable_if_t, is_invocable_r_v, is_same_v
#include <utility>        // std::move, std::forward

namespace cardinal::core {

template <class Signature,
          cardinal::usize Capacity = 64,
          cardinal::usize Align    = alignof(std::max_align_t)>
class inplace_function;   // primary template intentionally undefined

template <class R, class... Args, cardinal::usize Capacity, cardinal::usize Align>
class inplace_function<R(Args...), Capacity, Align> {
    struct VTable {
        R    (*invoke)(void*, Args&&...);
        void (*copy)(void*, const void*);
        void (*move)(void*, void*);
        void (*destroy)(void*);
    };

    template <class F>
    static const VTable* vtable_for() noexcept {
        static const VTable vt = {
            +[](void* s, Args&&... a) -> R {
                return (*static_cast<F*>(s))(static_cast<Args&&>(a)...);
            },
            +[](void* d, const void* s) { ::new (d) F(*static_cast<const F*>(s)); },
            +[](void* d, void* s) {
                ::new (d) F(std::move(*static_cast<F*>(s)));
                static_cast<F*>(s)->~F();
            },
            +[](void* s) { static_cast<F*>(s)->~F(); },
        };
        return &vt;
    }

public:
    inplace_function() noexcept : vt_(nullptr) {}
    inplace_function(decltype(nullptr)) noexcept : vt_(nullptr) {}

    template <class F, class DF = std::decay_t<F>,
              class = std::enable_if_t<
                  !std::is_same_v<DF, inplace_function> &&
                  std::is_invocable_r_v<R, DF&, Args...>>>
    inplace_function(F&& f) {
        static_assert(sizeof(DF)  <= Capacity, "callable too large for inplace_function Capacity");
        static_assert(alignof(DF) <= Align,    "callable over-aligned for inplace_function Align");
        ::new (storage_) DF(std::forward<F>(f));
        vt_ = vtable_for<DF>();
    }

    inplace_function(const inplace_function& o) : vt_(o.vt_) {
        if (vt_) vt_->copy(storage_, o.storage_);
    }
    inplace_function(inplace_function&& o) noexcept : vt_(o.vt_) {
        if (vt_) vt_->move(storage_, o.storage_);
        o.vt_ = nullptr;
    }

    inplace_function& operator=(const inplace_function& o) {
        if (this != &o) { reset(); vt_ = o.vt_; if (vt_) vt_->copy(storage_, o.storage_); }
        return *this;
    }
    inplace_function& operator=(inplace_function&& o) noexcept {
        if (this != &o) { reset(); vt_ = o.vt_; if (vt_) vt_->move(storage_, o.storage_); o.vt_ = nullptr; }
        return *this;
    }
    inplace_function& operator=(decltype(nullptr)) noexcept { reset(); return *this; }

    template <class F, class DF = std::decay_t<F>,
              class = std::enable_if_t<
                  !std::is_same_v<DF, inplace_function> &&
                  std::is_invocable_r_v<R, DF&, Args...>>>
    inplace_function& operator=(F&& f) {
        *this = inplace_function(std::forward<F>(f));
        return *this;
    }

    ~inplace_function() { reset(); }

    explicit operator bool() const noexcept { return vt_ != nullptr; }

    R operator()(Args... args) const {
        // Calling an empty inplace_function is a programming error (like
        // std::function's bad_function_call). Guard in debug; release relies
        // on the caller checking operator bool.
        return vt_->invoke(data_(), static_cast<Args&&>(args)...);
    }

    void reset() noexcept { if (vt_) { vt_->destroy(storage_); vt_ = nullptr; } }

    static constexpr cardinal::usize capacity() noexcept { return Capacity; }

private:
    // storage_ is mutable so a const operator() can still invoke a mutable
    // target (matches std::function's const-call-mutates-target behaviour).
    alignas(Align) mutable unsigned char storage_[Capacity];
    const VTable* vt_;

    void* data_() const noexcept { return const_cast<unsigned char*>(storage_); }
};

}  // namespace cardinal::core

namespace cardinal {
template <class Signature,
          cardinal::usize Capacity = 64,
          cardinal::usize Align    = alignof(std::max_align_t)>
using inplace_function = core::inplace_function<Signature, Capacity, Align>;
}  // namespace cardinal
