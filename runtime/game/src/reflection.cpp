#include <cardinal/game/reflection.hpp>

#include <cardinal/core/algorithm.hpp>
#include <cardinal/core/utility.hpp>

namespace cardinal::game {

ClassRegistry& ClassRegistry::instance() {
    // Process-singleton with deterministic destruction (order-of-static-init
    // safe because the macro registrations write into this on first use).
    static ClassRegistry s;
    return s;
}

void ClassRegistry::register_class(ClassDef def) {
    // Last-wins on duplicate name (matches the cppscript reload pattern).
    for (auto& c : classes_) {
        if (c.name == def.name) { c = cardinal::move(def); return; }
    }
    classes_.push_back(cardinal::move(def));
}

const ClassDef* ClassRegistry::find(const cardinal::string& name) const {
    for (const auto& c : classes_) if (c.name == name) return &c;
    return nullptr;
}

cardinal::vector<cardinal::string> ClassRegistry::all_names() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(classes_.size());
    for (const auto& c : classes_) r.push_back(c.name);
    cardinal::sort(r.begin(), r.end());
    return r;
}

cardinal::vector<const ClassDef*> ClassRegistry::all_in_category(
    const cardinal::string& cat_prefix) const
{
    cardinal::vector<const ClassDef*> r;
    for (const auto& c : classes_) {
        if (c.category.compare(0, cat_prefix.size(), cat_prefix) == 0) {
            r.push_back(&c);
        }
    }
    cardinal::sort(r.begin(), r.end(),
        [](const ClassDef* a, const ClassDef* b){
            if (a->category != b->category) return a->category < b->category;
            return a->name < b->name;
        });
    return r;
}

usize ClassRegistry::size() const noexcept { return classes_.size(); }

}  // namespace cardinal::game
