#pragma once

// =============================================================================
// cardinal::core::DenseMap<K, V> — open-addressing hash map (linear probing,
// power-of-two buckets, tombstone deletion, 0.7 max load factor).
//
// Complements flat_map (sorted vector, O(log N) find, ordered iteration):
// DenseMap is O(1) average find/insert/erase with a single flat bucket array,
// for the hot per-frame lookups flat_map's log-N + the std::unordered_map
// node-per-element pointer chasing both lose on — id→record tables, command
// registries (StringId→Command), entity→index maps, resource caches.
//
// Constraints (v1): K and V must be default-constructible (empty/tombstone
// slots hold value-initialised K/V) and K copyable. Iteration order is
// unspecified + unstable across rehash. For ordered iteration use flat_map.
//
// FOUNDATION RULE: lives in cardinal::core (uses std + core containers).
// Exposed as cardinal::dense_map.
// =============================================================================

#include <cardinal/core/types.hpp>        // u8, usize, cardinal::hash
#include <cardinal/core/containers.hpp>   // cardinal::vector
#include <cardinal/core/utility.hpp>      // cardinal::move, cardinal::pair

#include <functional>     // std::equal_to

namespace cardinal::core {

template <class K, class V,
          class Hash  = cardinal::hash<K>,
          class KeyEq = std::equal_to<K>>
class DenseMap {
    enum class State : cardinal::u8 { Empty = 0, Filled = 1, Tomb = 2 };
    struct Slot {
        State state{State::Empty};
        K     key{};
        V     value{};
    };

public:
    using key_type    = K;
    using mapped_type = V;
    using size_type   = cardinal::usize;

    DenseMap() = default;
    explicit DenseMap(size_type capacity_hint) { reserve(capacity_hint); }

    // ---- capacity -----------------------------------------------------
    [[nodiscard]] bool empty()    const noexcept { return size_ == 0; }
    size_type          size()     const noexcept { return size_; }
    size_type          capacity() const noexcept { return slots_.size(); }

    void clear() noexcept {
        for (Slot& s : slots_) { s.state = State::Empty; s.key = K{}; s.value = V{}; }
        size_ = 0;
        tombstones_ = 0;
    }

    // Ensure room for at least n elements without a rehash.
    void reserve(size_type n) {
        const size_type need = min_buckets_for_(n);
        if (need > slots_.size()) rehash_(need);
    }

    // ---- lookup -------------------------------------------------------
    V* find(const K& k) noexcept {
        const size_type idx = find_index_(k);
        return idx == npos ? nullptr : &slots_[idx].value;
    }
    const V* find(const K& k) const noexcept {
        const size_type idx = find_index_(k);
        return idx == npos ? nullptr : &slots_[idx].value;
    }
    bool contains(const K& k) const noexcept { return find_index_(k) != npos; }

    // ---- insertion ----------------------------------------------------
    // Returns {pointer-to-value, inserted?}. If the key already exists the
    // existing value is left unchanged (use operator[]/insert_or_assign to
    // overwrite).
    cardinal::pair<V*, bool> insert(const K& k, const V& v) {
        return emplace_(k, v);
    }
    cardinal::pair<V*, bool> insert(const K& k, V&& v) {
        return emplace_(k, cardinal::move(v));
    }

    V& insert_or_assign(const K& k, const V& v) {
        auto [p, inserted] = emplace_(k, v);
        if (!inserted) *p = v;
        return *p;
    }

    // Default-construct (or return existing) value for key k.
    V& operator[](const K& k) {
        return *emplace_(k, V{}).first;
    }

    // ---- erase --------------------------------------------------------
    bool erase(const K& k) noexcept {
        const size_type idx = find_index_(k);
        if (idx == npos) return false;
        slots_[idx].state = State::Tomb;
        slots_[idx].key   = K{};
        slots_[idx].value = V{};
        --size_;
        ++tombstones_;
        return true;
    }

    // ---- iteration ----------------------------------------------------
    // Visit every live entry. fn is invoked as fn(const K&, V&).
    template <class Fn>
    void for_each(Fn&& fn) {
        for (Slot& s : slots_)
            if (s.state == State::Filled) fn(static_cast<const K&>(s.key), s.value);
    }
    template <class Fn>
    void for_each(Fn&& fn) const {
        for (const Slot& s : slots_)
            if (s.state == State::Filled) fn(s.key, s.value);
    }

    // Minimal forward iterator (skips empty/tombstone slots). Dereferences to
    // a {first, second} proxy so structured bindings work:
    //   for (auto [k, v] : map) ...
    class iterator {
    public:
        struct Ref { const K& first; V& second; };
        iterator(Slot* p, Slot* end) noexcept : p_(p), end_(end) { skip_(); }
        Ref operator*()  const noexcept { return Ref{ p_->key, p_->value }; }
        iterator& operator++() noexcept { ++p_; skip_(); return *this; }
        bool operator==(const iterator& o) const noexcept { return p_ == o.p_; }
        bool operator!=(const iterator& o) const noexcept { return p_ != o.p_; }
    private:
        void skip_() noexcept { while (p_ != end_ && p_->state != State::Filled) ++p_; }
        Slot* p_; Slot* end_;
    };
    iterator begin() noexcept {
        Slot* d = slots_.data();
        return iterator(d, d + slots_.size());
    }
    iterator end() noexcept {
        Slot* d = slots_.data();
        return iterator(d + slots_.size(), d + slots_.size());
    }

private:
    static constexpr size_type npos = static_cast<size_type>(-1);

    cardinal::vector<Slot> slots_;
    size_type              size_{0};
    size_type              tombstones_{0};
    Hash                   hasher_{};
    KeyEq                  eq_{};

    // Smallest power-of-two bucket count that holds n elements under the 0.7
    // load factor (min 8). load 0.7 => need = ceil(n / 0.7) rounded up to pow2.
    static size_type min_buckets_for_(size_type n) noexcept {
        size_type need = 8;
        // n*10/7 is the buckets needed at 70% load; grow pow2 until it fits.
        const size_type target = (n == 0) ? 1 : (n * 10u / 7u + 1u);
        while (need < target) need <<= 1;
        return need;
    }

    size_type find_index_(const K& k) const noexcept {
        if (slots_.empty()) return npos;
        const size_type mask = slots_.size() - 1;
        size_type i = static_cast<size_type>(hasher_(k)) & mask;
        for (size_type probe = 0; probe <= mask; ++probe) {
            const Slot& s = slots_[i];
            if (s.state == State::Empty) return npos;           // probe chain ended
            if (s.state == State::Filled && eq_(s.key, k)) return i;
            i = (i + 1) & mask;                                 // tombstone or mismatch: keep going
        }
        return npos;
    }

    template <class VV>
    cardinal::pair<V*, bool> emplace_(const K& k, VV&& v) {
        ensure_capacity_for_one_();
        const size_type mask = slots_.size() - 1;
        size_type i = static_cast<size_type>(hasher_(k)) & mask;
        size_type first_tomb = npos;
        for (size_type probe = 0; probe <= mask; ++probe) {
            Slot& s = slots_[i];
            if (s.state == State::Empty) {
                // Reuse the earliest tombstone seen on the chain if any.
                Slot& dst = (first_tomb != npos) ? slots_[first_tomb] : s;
                if (first_tomb != npos) --tombstones_;
                dst.state = State::Filled;
                dst.key   = k;
                dst.value = static_cast<VV&&>(v);
                ++size_;
                return { &dst.value, true };
            }
            if (s.state == State::Filled && eq_(s.key, k)) {
                return { &s.value, false };                     // already present
            }
            if (s.state == State::Tomb && first_tomb == npos) first_tomb = i;
            i = (i + 1) & mask;
        }
        // Table full of filled+tombstones with no empty slot reached: rehash
        // (drops tombstones) and retry. Guaranteed to terminate.
        rehash_(slots_.size() << 1);
        return emplace_(k, static_cast<VV&&>(v));
    }

    void ensure_capacity_for_one_() {
        if (slots_.empty()) { rehash_(8); return; }
        // Rehash when live + tombstones cross 70% — keeps probe chains short.
        if ((size_ + tombstones_ + 1) * 10u >= slots_.size() * 7u)
            rehash_(slots_.size() << 1);
    }

    void rehash_(size_type new_cap) {
        cardinal::vector<Slot> old = cardinal::move(slots_);
        slots_.clear();
        slots_.resize(new_cap);             // value-inits all slots to Empty
        size_       = 0;
        tombstones_ = 0;
        const size_type mask = new_cap - 1;
        for (Slot& s : old) {
            if (s.state != State::Filled) continue;
            size_type i = static_cast<size_type>(hasher_(s.key)) & mask;
            while (slots_[i].state == State::Filled) i = (i + 1) & mask;
            slots_[i].state = State::Filled;
            slots_[i].key   = cardinal::move(s.key);
            slots_[i].value = cardinal::move(s.value);
            ++size_;
        }
    }
};

}  // namespace cardinal::core

namespace cardinal {
template <class K, class V,
          class Hash  = cardinal::hash<K>,
          class KeyEq = std::equal_to<K>>
using dense_map = core::DenseMap<K, V, Hash, KeyEq>;
}  // namespace cardinal
