#pragma once

// =============================================================================
// Cardinal — Result<T,E> sum type for error handling without exceptions.
//
// Modelled on Rust's Result. The engine compiles with /EHsc (exceptions on
// for STL compatibility) but doesn't *use* exceptions in its own code paths
// — too unpredictable for the per-frame budget and for cross-DLL boundaries.
// `Result<T,E>` is the in-house alternative.
//
//     Result<Mesh, LoadError> load_mesh(const char* path);
//
//     auto r = load_mesh("foo.obj");
//     if (r) { use(*r); } else { log_error(r.error()); }
//
// `T` may be void — `Result<void, E>` is a useful "operation succeeded /
// failed-with-reason" return.
// =============================================================================

#include <cardinal/core/types.hpp>

#include <new>            // placement new
#include <type_traits>
#include <utility>        // std::move / std::forward

namespace cardinal::core {

// Tag types for explicit construction.
struct ok_tag  {};
struct err_tag {};
inline constexpr ok_tag  ok{};
inline constexpr err_tag err{};

template <typename T, typename E>
class Result {
    static_assert(!std::is_reference_v<T>, "Result<T,E>: T must not be a reference");
    static_assert(!std::is_reference_v<E>, "Result<T,E>: E must not be a reference");
public:
    using value_type = T;
    using error_type = E;

    // ------ Construction ------
    template <typename... Args>
    Result(ok_tag, Args&&... a)  : has_(true)  { new (&v_.val) T(std::forward<Args>(a)...); }
    template <typename... Args>
    Result(err_tag, Args&&... a) : has_(false) { new (&v_.err) E(std::forward<Args>(a)...); }

    // Implicit "value-side" construction so a function can `return Mesh{...};`
    // without writing the tag every time. Disabled when T == E (ambiguous).
    template <typename U = T,
              typename = std::enable_if_t<std::is_constructible_v<T, U&&>
                                       && !std::is_same_v<std::decay_t<U>, Result>
                                       && !std::is_constructible_v<E, U&&>>>
    Result(U&& u) : has_(true) { new (&v_.val) T(std::forward<U>(u)); }

    Result(const Result& o)  : has_(o.has_) {
        if (has_) new (&v_.val) T(o.v_.val); else new (&v_.err) E(o.v_.err);
    }
    Result(Result&& o) noexcept(std::is_nothrow_move_constructible_v<T>
                              && std::is_nothrow_move_constructible_v<E>)
        : has_(o.has_)
    {
        if (has_) new (&v_.val) T(std::move(o.v_.val));
        else      new (&v_.err) E(std::move(o.v_.err));
    }
    Result& operator=(const Result& o) {
        if (this != &o) { this->~Result(); new (this) Result(o); }
        return *this;
    }
    Result& operator=(Result&& o) noexcept(std::is_nothrow_move_constructible_v<T>
                                         && std::is_nothrow_move_constructible_v<E>) {
        if (this != &o) { this->~Result(); new (this) Result(std::move(o)); }
        return *this;
    }
    ~Result() {
        if (has_) v_.val.~T(); else v_.err.~E();
    }

    // ------ Inspection ------
    bool is_ok()  const noexcept { return has_; }
    bool is_err() const noexcept { return !has_; }
    explicit operator bool() const noexcept { return has_; }

    T& value() &              { return v_.val; }
    const T& value() const &  { return v_.val; }
    T&& value() &&            { return std::move(v_.val); }

    E& error() &              { return v_.err; }
    const E& error() const &  { return v_.err; }
    E&& error() &&            { return std::move(v_.err); }

    T& operator*() &              { return v_.val; }
    const T& operator*() const &  { return v_.val; }
    T* operator->()               { return &v_.val; }
    const T* operator->() const   { return &v_.val; }

    // Return value or fall back. Doesn't take by ref&& — keeps the API
    // simple and matches std::optional<T>::value_or.
    template <typename U>
    T value_or(U&& fallback) const & {
        return has_ ? v_.val : static_cast<T>(std::forward<U>(fallback));
    }
    template <typename U>
    T value_or(U&& fallback) && {
        return has_ ? std::move(v_.val) : static_cast<T>(std::forward<U>(fallback));
    }

private:
    union Storage {
        T val;
        E err;
        Storage() {}
        ~Storage() {}
    } v_;
    bool has_;
};

// Specialisation for Result<void, E> — useful for "operation X succeeded /
// failed-with-reason" returns.
template <typename E>
class Result<void, E> {
public:
    using value_type = void;
    using error_type = E;

    Result(ok_tag) noexcept : has_(true) {}
    template <typename... Args>
    Result(err_tag, Args&&... a) : has_(false) { new (&err_) E(std::forward<Args>(a)...); }

    Result(const Result& o) : has_(o.has_) {
        if (!has_) new (&err_) E(o.err_);
    }
    Result(Result&& o) noexcept(std::is_nothrow_move_constructible_v<E>) : has_(o.has_) {
        if (!has_) new (&err_) E(std::move(o.err_));
    }
    Result& operator=(const Result& o) {
        if (this != &o) { this->~Result(); new (this) Result(o); }
        return *this;
    }
    Result& operator=(Result&& o) noexcept(std::is_nothrow_move_constructible_v<E>) {
        if (this != &o) { this->~Result(); new (this) Result(std::move(o)); }
        return *this;
    }
    ~Result() { if (!has_) err_.~E(); }

    bool is_ok()  const noexcept { return has_; }
    bool is_err() const noexcept { return !has_; }
    explicit operator bool() const noexcept { return has_; }

    E& error() &             { return err_; }
    const E& error() const & { return err_; }
    E&& error() &&           { return std::move(err_); }

private:
    union { E err_; };
    bool has_{true};
};

}  // namespace cardinal::core
