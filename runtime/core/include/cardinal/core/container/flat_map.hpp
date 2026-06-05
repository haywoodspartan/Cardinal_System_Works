#pragma once

// =============================================================================
// cardinal::core::flat_map<K, V, Cmp> — sorted-vector-backed associative
// map. Same semantics as std::map (unique keys, ordered iteration, O(log N)
// find) but stored as a single cardinal::vector<pair<K,V>> for cache-tight
// iteration and packed memory.
//
// See cardinal/core/flat_set.hpp for the cost model and design notes;
// flat_map adds the std::map key-value API: operator[], at, try_emplace,
// insert_or_assign, find/erase by key.
//
// Storage choice: single vector<pair<K, V>> (BinaryMap
// convention). The C++23 std::flat_map parallel-arrays layout (vector<K>
// + vector<V>) is faster for key-only scans but breaks API compat with
// std::map iterators — defer until a real consumer needs it.
//
// Iterators expose pair<K, V>& (NOT pair<const K, V>&) since the
// underlying storage is mutable for memmove/sort. Mutating the key from
// outside breaks the invariant — by convention, don't.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/containers.hpp>     // cardinal::vector
#include <cardinal/core/algorithm.hpp>      // cardinal::lower_bound / upper_bound
#include <cardinal/core/utility.hpp>        // cardinal::pair, move

#include <functional>      // std::less
#include <initializer_list>
#include <iterator>        // std::distance
#include <stdexcept>       // std::out_of_range — only thrown by at()

namespace cardinal::core {

template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<cardinal::pair<Key, T>>>
class flat_map {
public:
    using key_type        = Key;
    using mapped_type     = T;
    using value_type      = cardinal::pair<Key, T>;
    using key_compare     = Compare;
    using allocator_type  = Allocator;
    using container_type  = cardinal::vector<value_type, Allocator>;
    using size_type       = typename container_type::size_type;
    using difference_type = typename container_type::difference_type;
    using reference       = typename container_type::reference;
    using const_reference = typename container_type::const_reference;
    using pointer         = typename container_type::pointer;
    using const_pointer   = typename container_type::const_pointer;
    using iterator               = typename container_type::iterator;
    using const_iterator         = typename container_type::const_iterator;
    using reverse_iterator       = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    // Pair compare — orders pairs by their key only.
    struct value_compare {
        [[no_unique_address]] Compare cmp;
        [[nodiscard]] bool operator()(const value_type& a, const value_type& b) const noexcept { return cmp(a.first, b.first); }
        [[nodiscard]] bool operator()(const value_type& a, const Key& b)        const noexcept { return cmp(a.first, b); }
        [[nodiscard]] bool operator()(const Key& a,        const value_type& b) const noexcept { return cmp(a, b.first); }
    };

    // ---- ctors ---------------------------------------------------------
    flat_map() = default;
    explicit flat_map(const Compare& cmp, const Allocator& alloc = {}) noexcept
        : data_(alloc), vc_{cmp} {}
    explicit flat_map(const Allocator& alloc) noexcept : data_(alloc) {}

    template <class InputIt>
    flat_map(InputIt first, InputIt last,
             const Compare& cmp = {}, const Allocator& alloc = {})
        : data_(alloc), vc_{cmp} {
        bulk_insert(first, last);
    }

    flat_map(std::initializer_list<value_type> init,
             const Compare& cmp = {}, const Allocator& alloc = {})
        : flat_map(init.begin(), init.end(), cmp, alloc) {}

    // ---- size / capacity ----------------------------------------------
    [[nodiscard]] bool      empty()    const noexcept { return data_.empty(); }
    [[nodiscard]] size_type size()     const noexcept { return data_.size(); }
    [[nodiscard]] size_type max_size() const noexcept { return data_.max_size(); }
    [[nodiscard]] size_type capacity() const noexcept { return data_.capacity(); }

    void reserve(size_type n)            { data_.reserve(n); }
    void shrink_to_fit()                 { data_.shrink_to_fit(); }
    void clear()                noexcept { data_.clear(); }

    // ---- iterators -----------------------------------------------------
    [[nodiscard]] iterator               begin()        noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator         begin()  const noexcept { return data_.begin(); }
    [[nodiscard]] const_iterator         cbegin() const noexcept { return data_.cbegin(); }
    [[nodiscard]] iterator               end()          noexcept { return data_.end(); }
    [[nodiscard]] const_iterator         end()    const noexcept { return data_.end(); }
    [[nodiscard]] const_iterator         cend()   const noexcept { return data_.cend(); }

    [[nodiscard]] reverse_iterator       rbegin()        noexcept { return data_.rbegin(); }
    [[nodiscard]] const_reverse_iterator rbegin()  const noexcept { return data_.rbegin(); }
    [[nodiscard]] const_reverse_iterator crbegin() const noexcept { return data_.crbegin(); }
    [[nodiscard]] reverse_iterator       rend()          noexcept { return data_.rend(); }
    [[nodiscard]] const_reverse_iterator rend()    const noexcept { return data_.rend(); }
    [[nodiscard]] const_reverse_iterator crend()   const noexcept { return data_.crend(); }

    // ---- lookup --------------------------------------------------------
    [[nodiscard]] iterator       lower_bound(const Key& k)       noexcept { return cardinal::lower_bound(data_.begin(), data_.end(), k, vc_); }
    [[nodiscard]] const_iterator lower_bound(const Key& k) const noexcept { return cardinal::lower_bound(data_.begin(), data_.end(), k, vc_); }
    [[nodiscard]] iterator       upper_bound(const Key& k)       noexcept { return cardinal::upper_bound(data_.begin(), data_.end(), k, vc_); }
    [[nodiscard]] const_iterator upper_bound(const Key& k) const noexcept { return cardinal::upper_bound(data_.begin(), data_.end(), k, vc_); }

    [[nodiscard]] iterator find(const Key& k) noexcept {
        const iterator it = lower_bound(k);
        return (it != data_.end() && !vc_.cmp(k, it->first)) ? it : data_.end();
    }
    [[nodiscard]] const_iterator find(const Key& k) const noexcept {
        const const_iterator it = lower_bound(k);
        return (it != data_.end() && !vc_.cmp(k, it->first)) ? it : data_.end();
    }
    [[nodiscard]] bool      contains(const Key& k) const noexcept { return find(k) != data_.end(); }
    [[nodiscard]] size_type count   (const Key& k) const noexcept { return contains(k) ? 1u : 0u; }

    [[nodiscard]] cardinal::pair<iterator, iterator>
    equal_range(const Key& k) noexcept {
        const iterator lb = lower_bound(k);
        return {lb, (lb != data_.end() && !vc_.cmp(k, lb->first)) ? std::next(lb) : lb};
    }
    [[nodiscard]] cardinal::pair<const_iterator, const_iterator>
    equal_range(const Key& k) const noexcept {
        const const_iterator lb = lower_bound(k);
        return {lb, (lb != data_.end() && !vc_.cmp(k, lb->first)) ? std::next(lb) : lb};
    }

    // ---- element access ------------------------------------------------
    T& at(const Key& k) {
        const iterator it = find(k);
        if (it == data_.end()) throw std::out_of_range("cardinal::flat_map::at");
        return it->second;
    }
    const T& at(const Key& k) const {
        const const_iterator it = find(k);
        if (it == data_.end()) throw std::out_of_range("cardinal::flat_map::at");
        return it->second;
    }
    T& operator[](const Key& k) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) return lb->second;
        return data_.insert(lb, value_type(k, T{}))->second;
    }
    T& operator[](Key&& k) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) return lb->second;
        return data_.insert(lb, value_type(std::move(k), T{}))->second;
    }

    // ---- insert / emplace ---------------------------------------------
    cardinal::pair<iterator, bool> insert(const value_type& v) {
        const iterator lb = lower_bound(v.first);
        if (lb != data_.end() && !vc_.cmp(v.first, lb->first)) return {lb, false};
        return {data_.insert(lb, v), true};
    }
    cardinal::pair<iterator, bool> insert(value_type&& v) {
        const iterator lb = lower_bound(v.first);
        if (lb != data_.end() && !vc_.cmp(v.first, lb->first)) return {lb, false};
        return {data_.insert(lb, std::move(v)), true};
    }

    template <class... Args>
    cardinal::pair<iterator, bool> emplace(Args&&... args) {
        value_type tmp(std::forward<Args>(args)...);
        const iterator lb = lower_bound(tmp.first);
        if (lb != data_.end() && !vc_.cmp(tmp.first, lb->first)) return {lb, false};
        return {data_.insert(lb, std::move(tmp)), true};
    }

    template <class... Args>
    cardinal::pair<iterator, bool> try_emplace(const Key& k, Args&&... args) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) return {lb, false};
        return {data_.insert(lb, value_type(k, T(std::forward<Args>(args)...))), true};
    }
    template <class... Args>
    cardinal::pair<iterator, bool> try_emplace(Key&& k, Args&&... args) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) return {lb, false};
        return {data_.insert(lb, value_type(std::move(k), T(std::forward<Args>(args)...))), true};
    }

    template <class M>
    cardinal::pair<iterator, bool> insert_or_assign(const Key& k, M&& v) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) {
            lb->second = std::forward<M>(v);
            return {lb, false};
        }
        return {data_.insert(lb, value_type(k, std::forward<M>(v))), true};
    }
    template <class M>
    cardinal::pair<iterator, bool> insert_or_assign(Key&& k, M&& v) {
        const iterator lb = lower_bound(k);
        if (lb != data_.end() && !vc_.cmp(k, lb->first)) {
            lb->second = std::forward<M>(v);
            return {lb, false};
        }
        return {data_.insert(lb, value_type(std::move(k), std::forward<M>(v))), true};
    }

    // Range insert — goes through bulk_insert path.
    template <class InputIt>
    void insert(InputIt first, InputIt last) { bulk_insert(first, last); }

    void insert(std::initializer_list<value_type> ilist) {
        bulk_insert(ilist.begin(), ilist.end());
    }

    // Bulk insert — append + sort + in-place-merge + dedup. O((M+N) log N).
    // Amortised better than M single inserts whenever N > log M.
    template <class InputIt>
    void bulk_insert(InputIt first, InputIt last) {
        const size_type old_n = data_.size();
        const auto in_n = static_cast<size_type>(std::distance(first, last));
        if (in_n == 0) return;
        data_.reserve(old_n + in_n);
        for (auto it = first; it != last; ++it) data_.push_back(*it);

        // Sort the tail by key.
        std::sort(data_.begin() + static_cast<difference_type>(old_n),
                  data_.end(), vc_);
        // Merge sorted halves.
        std::inplace_merge(data_.begin(),
                           data_.begin() + static_cast<difference_type>(old_n),
                           data_.end(), vc_);
        // Deduplicate by key — keep the FIRST occurrence per key, drop later
        // ones. (std::unique keeps the first of each equal-run, which is
        // the std::map "insert only if not present" semantic.)
        data_.erase(std::unique(data_.begin(), data_.end(),
                                [this](const value_type& a, const value_type& b) noexcept {
                                    return !vc_.cmp(a.first, b.first) && !vc_.cmp(b.first, a.first);
                                }),
                    data_.end());
    }

    // ---- erase ---------------------------------------------------------
    iterator  erase(const_iterator pos)                          { return data_.erase(pos); }
    iterator  erase(const_iterator first, const_iterator last)   { return data_.erase(first, last); }
    size_type erase(const Key& k) {
        const iterator it = find(k);
        if (it == data_.end()) return 0;
        data_.erase(it);
        return 1;
    }

    void swap(flat_map& other) noexcept {
        using std::swap;
        data_.swap(other.data_);
        swap(vc_, other.vc_);
    }

    [[nodiscard]] const container_type& container() const noexcept { return data_; }
    [[nodiscard]] container_type&       container()       noexcept { return data_; }

    [[nodiscard]] key_compare   key_comp()   const noexcept { return vc_.cmp; }
    [[nodiscard]] value_compare value_comp() const noexcept { return vc_; }

    friend bool operator==(const flat_map& a, const flat_map& b) noexcept { return a.data_ == b.data_; }
    friend bool operator!=(const flat_map& a, const flat_map& b) noexcept { return a.data_ != b.data_; }

private:
    container_type                  data_;
    [[no_unique_address]] value_compare vc_{};
};

template <class K, class V, class C, class A>
void swap(flat_map<K, V, C, A>& a, flat_map<K, V, C, A>& b) noexcept { a.swap(b); }

}  // namespace cardinal::core
