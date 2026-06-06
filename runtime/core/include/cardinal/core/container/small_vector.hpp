#pragma once

// =============================================================================
// cardinal::core::SmallVector<T, N> — a dynamic array with N elements of
// inline (small-buffer) storage. No heap allocation until size() exceeds N;
// past that it grows on the heap exactly like cardinal::vector. This is the
// engine's go-to container for the many SMALL, short-lived arrays in hot
// paths — per-entity child lists, per-pass resource binds, command argv,
// draw-batch scratch — where the typical count is a handful and a heap
// allocation per use would dominate.
//
// API mirrors the std::vector subset Cardinal actually uses (push/emplace
// back, pop, resize, reserve, clear, erase, indexed + iterator access,
// front/back/data, copy/move, initializer_list, ==). It is NOT a drop-in for
// every std::vector method — by design; add members when a real call site
// needs one.
//
// Lifetime: elements are placement-new constructed and explicitly destroyed;
// the inline buffer is a raw aligned byte array, so T need not be default-
// constructible to declare a SmallVector (only to resize(n) without a value).
//
// FOUNDATION RULE: this lives in cardinal::core, so it may use std headers
// directly. Non-core code uses the cardinal::small_vector alias (bottom).
// =============================================================================

#include <cardinal/core/types.hpp>   // cardinal::usize

#include <initializer_list>
#include <new>            // placement new, ::operator new/delete
#include <stdexcept>      // std::out_of_range (at())
#include <type_traits>    // std::is_nothrow_move_constructible
#include <utility>        // std::move, std::forward

namespace cardinal::core {

template <class T, cardinal::usize N>
class SmallVector {
    static_assert(N >= 1, "SmallVector inline capacity N must be >= 1");

public:
    using value_type      = T;
    using size_type       = cardinal::usize;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    SmallVector() noexcept : data_(inline_ptr()), size_(0), cap_(N) {}

    SmallVector(std::initializer_list<T> il) : SmallVector() {
        reserve(il.size());
        for (const T& v : il) emplace_back(v);
    }

    // Fill constructor.
    SmallVector(size_type count, const T& value) : SmallVector() {
        reserve(count);
        for (size_type i = 0; i < count; ++i) emplace_back(value);
    }

    SmallVector(const SmallVector& other) : SmallVector() {
        reserve(other.size_);
        for (size_type i = 0; i < other.size_; ++i) emplace_back(other.data_[i]);
    }

    SmallVector(SmallVector&& other) noexcept : SmallVector() {
        move_from_(std::move(other));
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear();
            reserve(other.size_);
            for (size_type i = 0; i < other.size_; ++i) emplace_back(other.data_[i]);
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            reset_();              // destroy ours + free our heap; back to empty-inline
            move_from_(std::move(other));
        }
        return *this;
    }

    ~SmallVector() { reset_(); }

    // ---- element access ------------------------------------------------
    reference       operator[](size_type i)       noexcept { return data_[i]; }
    const_reference operator[](size_type i) const noexcept { return data_[i]; }

    reference at(size_type i) {
        if (i >= size_) throw std::out_of_range("SmallVector::at");
        return data_[i];
    }
    const_reference at(size_type i) const {
        if (i >= size_) throw std::out_of_range("SmallVector::at");
        return data_[i];
    }

    reference       front()       noexcept { return data_[0]; }
    const_reference front() const noexcept { return data_[0]; }
    reference       back()        noexcept { return data_[size_ - 1]; }
    const_reference back()  const noexcept { return data_[size_ - 1]; }

    pointer       data()       noexcept { return data_; }
    const_pointer data() const noexcept { return data_; }

    // ---- iterators ----------------------------------------------------
    iterator       begin()       noexcept { return data_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator cbegin() const noexcept { return data_; }
    iterator       end()         noexcept { return data_ + size_; }
    const_iterator end()   const noexcept { return data_ + size_; }
    const_iterator cend()  const noexcept { return data_ + size_; }

    // ---- capacity -----------------------------------------------------
    [[nodiscard]] bool empty()  const noexcept { return size_ == 0; }
    size_type          size()   const noexcept { return size_; }
    size_type          capacity() const noexcept { return cap_; }
    // True while still using the inline buffer (no heap allocation yet).
    bool               is_inline() const noexcept { return data_ == inline_ptr(); }
    static constexpr size_type inline_capacity() noexcept { return N; }

    void reserve(size_type n) { if (n > cap_) grow_to_(n); }

    // ---- modifiers ----------------------------------------------------
    void clear() noexcept {
        destroy_range_(data_, size_);
        size_ = 0;
    }

    template <class... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == cap_) grow_to_(cap_ < N ? N + 1 : cap_ * 2);
        T* p = data_ + size_;
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
        ++size_;
        return *p;
    }

    void push_back(const T& v) { emplace_back(v); }
    void push_back(T&& v)      { emplace_back(std::move(v)); }

    void pop_back() noexcept {
        --size_;
        data_[size_].~T();
    }

    void resize(size_type n) {
        if (n < size_) { destroy_range_(data_ + n, size_ - n); size_ = n; }
        else { reserve(n); while (size_ < n) emplace_back(); }
    }
    void resize(size_type n, const T& value) {
        if (n < size_) { destroy_range_(data_ + n, size_ - n); size_ = n; }
        else { reserve(n); while (size_ < n) push_back(value); }
    }

    void assign(size_type count, const T& value) {
        clear();
        reserve(count);
        for (size_type i = 0; i < count; ++i) push_back(value);
    }

    // Erase one element; later elements shift down by one. Returns an
    // iterator to the element that followed the erased one.
    iterator erase(iterator pos) {
        for (iterator it = pos, nx = pos + 1; nx != end(); ++it, ++nx)
            *it = std::move(*nx);
        pop_back();
        return pos;
    }

    void swap(SmallVector& other) noexcept {
        // Correct regardless of inline/heap state on either side. Heap/heap
        // is the only case where pointer-swap would beat element moves, but
        // small-N is the design point, so keep it simple + always correct.
        SmallVector tmp(std::move(*this));
        *this = std::move(other);
        other = std::move(tmp);
    }

    // ---- comparison ---------------------------------------------------
    friend bool operator==(const SmallVector& a, const SmallVector& b) {
        if (a.size_ != b.size_) return false;
        for (size_type i = 0; i < a.size_; ++i)
            if (!(a.data_[i] == b.data_[i])) return false;
        return true;
    }
    friend bool operator!=(const SmallVector& a, const SmallVector& b) {
        return !(a == b);
    }

private:
    alignas(T) unsigned char inline_storage_[sizeof(T) * N];
    T*        data_;
    size_type size_;
    size_type cap_;

    T*       inline_ptr()       noexcept { return reinterpret_cast<T*>(inline_storage_); }
    const T* inline_ptr() const noexcept { return reinterpret_cast<const T*>(inline_storage_); }

    static void destroy_range_(T* p, size_type n) noexcept {
        for (size_type i = 0; i < n; ++i) p[i].~T();
    }

    // Grow capacity to at least new_cap, moving existing elements into a
    // fresh heap block (the inline buffer can only ever be the first home).
    void grow_to_(size_type new_cap) {
        if (new_cap <= cap_) return;
        T* new_data = static_cast<T*>(::operator new(sizeof(T) * new_cap));
        for (size_type i = 0; i < size_; ++i) {
            ::new (static_cast<void*>(new_data + i)) T(std::move(data_[i]));
            data_[i].~T();
        }
        if (!is_inline()) ::operator delete(data_);
        data_ = new_data;
        cap_  = new_cap;
    }

    // Destroy all elements, free the heap block (if any), return to the
    // empty inline state. Safe to call repeatedly.
    void reset_() noexcept {
        destroy_range_(data_, size_);
        if (!is_inline()) ::operator delete(data_);
        data_ = inline_ptr();
        size_ = 0;
        cap_  = N;
    }

    // Precondition: *this is freshly empty-inline (a default/after-reset_).
    void move_from_(SmallVector&& o) noexcept {
        if (o.is_inline()) {
            for (size_type i = 0; i < o.size_; ++i)
                ::new (static_cast<void*>(inline_ptr() + i)) T(std::move(o.data_[i]));
            size_ = o.size_;
            o.clear();             // destroy o's moved-from elements; o stays inline-empty
        } else {
            data_ = o.data_;       // steal the heap block
            size_ = o.size_;
            cap_  = o.cap_;
            o.data_ = o.inline_ptr();   // hand o back its (empty) inline storage
            o.size_ = 0;
            o.cap_  = N;
        }
    }
};

template <class T, cardinal::usize N>
inline void swap(SmallVector<T, N>& a, SmallVector<T, N>& b) noexcept { a.swap(b); }

}  // namespace cardinal::core

// Non-core call sites use this alias (mirrors cardinal::vector etc.).
namespace cardinal {
template <class T, cardinal::usize N>
using small_vector = core::SmallVector<T, N>;
}  // namespace cardinal
