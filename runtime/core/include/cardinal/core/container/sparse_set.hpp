#pragma once

// =============================================================================
// cardinal::core::SparseSet<T, Key> — sparse-to-dense associative storage.
//
// The classic ECS / system-component container: an unsigned integer Key (an
// entity / actor / resource id) maps to a value, with O(1) insert / remove /
// contains / get AND a PACKED dense array of values for cache-tight iteration
// (no holes to skip, unlike a hash map). Two index arrays:
//   sparse_[key]  -> position in the dense arrays (or kInvalid)
//   dense_keys_[] -> packed keys      (parallel to dense_values_)
//   dense_values_[] -> packed values
// Removal is swap-with-last + pop, so the dense arrays stay contiguous.
//
// Cost model vs the engine's other maps:
//   - O(1) everything + the fastest iteration (flat values, no skipping), but
//   - sparse_ is sized to (max key + 1): ideal for DENSE / bounded id spaces
//     (entity ids 0..N). For large, sparse, or non-integer keys use DenseMap
//     (hashed) or flat_map (ordered) instead.
// T need NOT be default-constructible (values are emplaced, never value-init).
//
// FOUNDATION RULE: lives in cardinal::core. Exposed as cardinal::sparse_set.
// =============================================================================

#include <cardinal/core/types.hpp>        // u32, usize
#include <cardinal/core/containers.hpp>   // cardinal::vector / span
#include <cardinal/core/utility.hpp>      // cardinal::move

#include <type_traits>    // std::is_unsigned_v
#include <utility>        // std::forward

namespace cardinal::core {

template <class T, class Key = cardinal::u32>
class SparseSet {
    static_assert(std::is_unsigned_v<Key>, "SparseSet Key must be an unsigned integer");

public:
    using key_type    = Key;
    using value_type  = T;
    using size_type   = cardinal::usize;

    static constexpr Key kInvalid = static_cast<Key>(-1);

    // ---- capacity -----------------------------------------------------
    [[nodiscard]] bool empty()    const noexcept { return dense_keys_.empty(); }
    size_type          size()     const noexcept { return dense_keys_.size(); }

    void reserve(size_type n) {
        dense_keys_.reserve(n);
        dense_values_.reserve(n);
    }

    void clear() noexcept {
        for (Key k : dense_keys_) sparse_[k] = kInvalid;   // reset only touched slots
        dense_keys_.clear();
        dense_values_.clear();
    }

    // ---- lookup -------------------------------------------------------
    [[nodiscard]] bool contains(Key k) const noexcept {
        return static_cast<size_type>(k) < sparse_.size()
            && sparse_[k] != kInvalid
            && static_cast<size_type>(sparse_[k]) < dense_keys_.size()
            && dense_keys_[sparse_[k]] == k;
    }
    T*       get(Key k)       noexcept { return contains(k) ? &dense_values_[sparse_[k]] : nullptr; }
    const T* get(Key k) const noexcept { return contains(k) ? &dense_values_[sparse_[k]] : nullptr; }

    // ---- insertion (overwrites an existing key's value) ---------------
    template <class... Args>
    T& emplace(Key k, Args&&... args) {
        ensure_sparse_(k);
        if (contains(k)) {
            T& slot = dense_values_[sparse_[k]];
            slot = T(std::forward<Args>(args)...);
            return slot;
        }
        sparse_[k] = static_cast<Key>(dense_keys_.size());
        dense_keys_.push_back(k);
        dense_values_.emplace_back(std::forward<Args>(args)...);
        return dense_values_.back();
    }
    T& insert(Key k, const T& v) { return emplace(k, v); }
    T& insert(Key k, T&& v)      { return emplace(k, cardinal::move(v)); }

    // ---- removal (swap-with-last + pop; keeps dense packed) -----------
    bool remove(Key k) {
        if (!contains(k)) return false;
        const Key       idx  = sparse_[k];
        const size_type last = dense_keys_.size() - 1;
        if (static_cast<size_type>(idx) != last) {
            dense_keys_[idx]   = dense_keys_[last];
            dense_values_[idx] = cardinal::move(dense_values_[last]);
            sparse_[dense_keys_[idx]] = idx;          // moved key's new home
        }
        dense_keys_.pop_back();
        dense_values_.pop_back();
        sparse_[k] = kInvalid;
        return true;
    }

    // ---- dense iteration ---------------------------------------------
    cardinal::span<const Key> keys()   const noexcept { return { dense_keys_.data(),   dense_keys_.size() }; }
    cardinal::span<T>         values()       noexcept { return { dense_values_.data(), dense_values_.size() }; }
    cardinal::span<const T>   values() const noexcept { return { dense_values_.data(), dense_values_.size() }; }

    template <class Fn> void for_each(Fn&& fn) {
        for (size_type i = 0; i < dense_keys_.size(); ++i) fn(dense_keys_[i], dense_values_[i]);
    }
    template <class Fn> void for_each(Fn&& fn) const {
        for (size_type i = 0; i < dense_keys_.size(); ++i) fn(dense_keys_[i], dense_values_[i]);
    }

    // Range-for over the packed values.
    auto begin()       noexcept { return dense_values_.begin(); }
    auto end()         noexcept { return dense_values_.end(); }
    auto begin() const noexcept { return dense_values_.begin(); }
    auto end()   const noexcept { return dense_values_.end(); }

private:
    cardinal::vector<Key> sparse_;
    cardinal::vector<Key> dense_keys_;
    cardinal::vector<T>   dense_values_;

    void ensure_sparse_(Key k) {
        if (static_cast<size_type>(k) >= sparse_.size())
            sparse_.resize(static_cast<size_type>(k) + 1, kInvalid);
    }
};

}  // namespace cardinal::core

namespace cardinal {
template <class T, class Key = cardinal::u32>
using sparse_set = core::SparseSet<T, Key>;
}  // namespace cardinal
