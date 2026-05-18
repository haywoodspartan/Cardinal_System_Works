#pragma once

// =============================================================================
// Cardinal core — container vocabulary.
//
// FOUNDATION RULE: the std container headers are referenced HERE and nowhere
// else outside cardinal::core. Non-core code uses cardinal::vector etc. so
// the allocator / storage policy stays restructurable from one place.
// Alias templates (not typedefs) so they remain fully generic.
// =============================================================================

#include <array>
#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <functional>   // std::less default for priority_queue
#include <span>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cardinal {

template <class T, class A = std::allocator<T>>
using vector = std::vector<T, A>;

template <class T, std::size_t N>
using array = std::array<T, N>;

template <class T, std::size_t E = std::dynamic_extent>
using span = std::span<T, E>;

template <class T, class A = std::allocator<T>>
using deque = std::deque<T, A>;

template <class T, class A = std::allocator<T>>
using list = std::list<T, A>;

template <class K, class V, class H = std::hash<K>,
          class Eq = std::equal_to<K>,
          class A  = std::allocator<std::pair<const K, V>>>
using unordered_map = std::unordered_map<K, V, H, Eq, A>;

template <class K, class H = std::hash<K>, class Eq = std::equal_to<K>,
          class A = std::allocator<K>>
using unordered_set = std::unordered_set<K, H, Eq, A>;

template <class K, class V, class C = std::less<K>,
          class A = std::allocator<std::pair<const K, V>>>
using map = std::map<K, V, C, A>;

template <class K, class C = std::less<K>, class A = std::allocator<K>>
using set = std::set<K, C, A>;

template <class T, class C = std::deque<T>>
using queue = std::queue<T, C>;

template <class T, class C = std::deque<T>>
using stack = std::stack<T, C>;

template <class T, class C = std::vector<T>,
          class Cmp = std::less<typename C::value_type>>
using priority_queue = std::priority_queue<T, C, Cmp>;

}  // namespace cardinal
