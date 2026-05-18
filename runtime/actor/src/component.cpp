#include <cardinal/actor/component.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::find/remove
#include <cardinal/core/utility.hpp>     // cardinal::move

namespace cardinal::actor {

cardinal::scene::Mat4 TransformComponent::matrix() const {
    using namespace cardinal::scene;
    return Mat4::translation(translation)
         * Mat4::rotation_xyz(rotation_euler)
         * Mat4::scaling(scale);
}

bool TagComponent::has(const cardinal::string& t) const noexcept {
    return cardinal::find(tags.begin(), tags.end(), t) != tags.end();
}

void TagComponent::add(cardinal::string t) {
    if (!has(t)) tags.push_back(cardinal::move(t));
}

void TagComponent::remove(const cardinal::string& t) {
    tags.erase(cardinal::remove(tags.begin(), tags.end(), t), tags.end());
}

}  // namespace cardinal::actor
