#include <cardinal/core/service/global_object_manager.hpp>

namespace cardinal::core {

GlobalObjectManager& GlobalObjectManager::instance() noexcept {
    // Function-local static — first call constructs, every subsequent call
    // returns the same instance. C++11 guarantees thread-safe init.
    static GlobalObjectManager mgr;
    return mgr;
}

void GlobalObjectManager::register_raw_(std::type_index type, std::string key,
                                        void* instance) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& e : entries_) {
        if (e.type == type && e.key == key) {
            // Overwrite the prior registration.
            e.instance = instance;
            return;
        }
    }
    entries_.push_back(Entry{type, std::move(key), instance});
}

void GlobalObjectManager::unregister_raw_(std::type_index type,
                                          const std::string& key) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->type == type && it->key == key) {
            entries_.erase(it);
            return;
        }
    }
}

void* GlobalObjectManager::get_raw_(std::type_index type,
                                    const std::string& key) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : entries_) {
        if (e.type == type && e.key == key) return e.instance;
    }
    return nullptr;
}

void GlobalObjectManager::clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

usize GlobalObjectManager::size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

}  // namespace cardinal::core
