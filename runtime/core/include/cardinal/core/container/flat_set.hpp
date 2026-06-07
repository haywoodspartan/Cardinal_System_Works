#pragma once

// =============================================================================
// cardinal::core::flat_set<T, Cmp> — sorted-vector-backed associative set.
//
// Same semantics as std::set: unique keys, ordered iteration, O(log N)
// find. Different cost model:
//
//   Operation        flat_set                 std::set (RB-tree)
//   find             O(log N) binary search   O(log N) tree traversal
//   insert (random)  O(N) — memmove tail      O(log N) rebalance
//   insert (sorted)  O(log N) amortised       O(log N)
//   erase            O(N) — memmove tail down O(log N)
//   iteration        contiguous, prefetchable each step pointer-chases
//   memory/elem      sizeof(T) packed         sizeof(T) + ~3*ptr + meta
//   cache misses     1 per find               log N per find
//
// Right fit: build-once-read-many tables (config, asset metadata, scene
// id→entity lookups), small/medium N (<10k), and any code path iterated
// frequently. Wrong fit: fast-churn workloads where insert/erase dominate.
//
// Modernisations vs. the sorted-vector reference:
//   * std::lower_bound — no `(first + last) / 2` overflow bug.
//   * Comparator stored as [[no_unique_address]] member (EBO for stateless
//     compares; the original reconstructed a fresh _COMPARE() on every loop iter).
//   * C++17/20 API: contains, emplace, equal_range, node-style insert hint,
//     bulk_insert(range) for amortised batch loads.
//   * Allocator-aware (defaults to cardinal::vector's allocator).
//   * Iterators are random-access (vector iterators), so users can do
//     std::ranges algorithms over the set directly.
// =============================================================================

#include <cardinal/core/types.hpp>
#include <cardinal/core/std/containers.hpp>     // cardinal::vector
#include <cardinal/core/std/algorithm.hpp>      // cardinal::lower_bound / upper_bound
#include <cardinal/core/std/utility.hpp>        // cardinal::pair

#include <functional>      // std::less
#include <initializer_list>
#include <iterator>        // std::distance, iterator_traits
#include <utility>         // std::move, std::forward

namespace cardinal::core {

template <class T, class Compare = std::less<T>,
          class Allocator = std::allocator<T>>
class flat_set {
public:
    using container_type         = cardinal::vector<T, Allocator>;
    using key_type               = T;
    using value_type             = T;
    using key_compare            = Compare;
    using value_compare          = Compare;
    using allocator_type         = Allocator;
    using size_type              = typename container_type::size_type;
    using difference_type        = typename container_type::difference_type;
    using reference              = typename container_type::reference;
    using const_reference        = typename container_type::const_reference;
    using pointer                = typename container_type::pointer;
    using const_pointer          = typename container_type::const_pointer;
    using iterator               = typename container_type::iterator;
    using const_iterator         = typename container_type::const_iterator;
    using reverse_iterator       = typename container_type::reverse_iterator;
    using const_reverse_iterator = typename container_type::const_reverse_iterator;

    // ---- ctors ---------------------------------------------------------
    flat_set() = default;
    explicit flat_set(const Compare& cmp, const Allocator& alloc = {}) noexcept
        : data_(alloc), cmp_(cmp) {}
    explicit flat_set(const Allocator& alloc) noexcept : data_(alloc) {}

    template <class InputIt>
    flat_set(InputIt first, InputIt last,
             const Compare& cmp = {}, const Allocator& alloc = {})
        : data_(alloc), cmp_(cmp) {
        bulk_insert(first, last);
    }

    flat_set(std::initializer_list<T> init,
             const Compare& cmp = {}, const Allocator& alloc = {})
        : flat_set(init.begin(), init.end(), cmp, alloc) {}

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
    [[nodiscard]] iterator       lower_bound(const T& key)       noexcept { return cardinal::lower_bound(data_.begin(), data_.end(), key, cmp_); }
    [[nodiscard]] const_iterator lower_bound(const T& key) const noexcept { return cardinal::lower_bound(data_.begin(), data_.end(), key, cmp_); }
    [[nodiscard]] iterator       upper_bound(const T& key)       noexcept { return cardinal::upper_bound(data_.begin(), data_.end(), key, cmp_); }
    [[nodiscard]] const_iterator upper_bound(const T& key) const noexcept { return cardinal::upper_bound(data_.begin(), data_.end(), key, cmp_); }

    [[nodiscard]] iterator find(const T& key) noexcept {
        const iterator it = lower_bound(key);
        return (it != data_.end() && !cmp_(key, *it)) ? it : data_.end();
    }
    [[nodiscard]] const_iterator find(const T& key) const noexcept {
        const const_iterator it = lower_bound(key);
        return (it != data_.end() && !cmp_(key, *it)) ? it : data_.end();
    }
    [[nodiscard]] bool      contains(const T& key) const noexcept { return find(key) != data_.end(); }
    [[nodiscard]] size_type count   (const T& key) const noexcept { return contains(key) ? 1u : 0u; }

    [[nodiscard]] cardinal::pair<iterator, iterator>
    equal_range(const T& key) noexcept {
        const iterator lb = lower_bound(key);
        return {lb, (lb != data_.end() && !cmp_(key, *lb)) ? std::next(lb) : lb};
    }
    [[nodiscard]] cardinal::pair<const_iterator, const_iterator>
    equal_range(const T& key) const noexcept {
        const const_iterator lb = lower_bound(key);
        return {lb, (lb != data_.end() && !cmp_(key, *lb)) ? std::next(lb) : lb};
    }

    // ---- insert / emplace ---------------------------------------------
    // Returns {iter, true} on insert, {existing_iter, false} on duplicate.
    cardinal::pair<iterator, bool> insert(const T& value) { return emplace(value); }
    cardinal::pair<iterator, bool> insert(T&& value)      { return emplace(std::move(value)); }

    template <class... Args>
    cardinal::pair<iterator, bool> emplace(Args&&... args) {
        T tmp(std::forward<Args>(args)...);
        const iterator lb = lower_bound(tmp);
        if (lb != data_.end() && !cmp_(tmp, *lb)) return {lb, false};
        return {data_.insert(lb, std::move(tmp)), true};
    }

    // Range insert — uses bulk_insert path; amortised better than N
    // single inserts when the input is large or already sorted.
    template <class InputIt>
    void insert(InputIt first, InputIt last) { bulk_insert(first, last); }

    void insert(std::initializer_list<T> ilist) {
        bulk_insert(ilist.begin(), ilist.end());
    }

    // Bulk insert — append then sort + unique-merge. O((M + N) log N) where
    // M = existing size, N = input range size. Beats M individual O(N)
    // inserts whenever N > log M.
    template <class InputIt>
    void bulk_insert(InputIt first, InputIt last) {
        const size_type old_n = data_.size();
        const auto in_n = static_cast<size_type>(std::distance(first, last));
        if (in_n == 0) return;
        data_.reserve(old_n + in_n);
        for (auto it = first; it != last; ++it) data_.push_back(*it);

        // Sort the newly-appended tail.
        std::sort(data_.begin() + static_cast<difference_type>(old_n),
                  data_.end(), cmp_);
        // Merge sorted halves in place (O(M + N)) — std::inplace_merge.
        std::inplace_merge(data_.begin(),
                           data_.begin() + static_cast<difference_type>(old_n),
                           data_.end(), cmp_);
        // Deduplicate adjacent equal keys.
        data_.erase(std::unique(data_.begin(), data_.end(),
                                [this](const T& a, const T& b) noexcept {
                                    return !cmp_(a, b) && !cmp_(b, a);
                                }),
                    data_.end());
    }

    // ---- erase ---------------------------------------------------------
    iterator  erase(const_iterator pos)                          { return data_.erase(pos); }
    iterator  erase(const_iterator first, const_iterator last)   { return data_.erase(first, last); }
    size_type erase(const T& key) {
        const iterator it = find(key);
        if (it == data_.end()) return 0;
        data_.erase(it);
        return 1;
    }

    void swap(flat_set& other) noexcept {
        using std::swap;
        data_.swap(other.data_);
        swap(cmp_, other.cmp_);
    }

    // ---- access to the raw vector (for serialisation, swap-in payloads etc.)
    [[nodiscard]] const container_type& container() const noexcept { return data_; }
    [[nodiscard]] container_type&       container()       noexcept { return data_; }

    // ---- comparators ---------------------------------------------------
    [[nodiscard]] key_compare   key_comp()   const noexcept { return cmp_; }
    [[nodiscard]] value_compare value_comp() const noexcept { return cmp_; }

    // ---- relational ops -----------------------------------------------
    friend bool operator==(const flat_set& a, const flat_set& b) noexcept { return a.data_ == b.data_; }
    friend bool operator!=(const flat_set& a, const flat_set& b) noexcept { return a.data_ != b.data_; }

private:
    container_type                data_;
    [[no_unique_address]] Compare cmp_{};
};

template <class T, class C, class A>
void swap(flat_set<T, C, A>& a, flat_set<T, C, A>& b) noexcept { a.swap(b); }

}  // namespace cardinal::core
