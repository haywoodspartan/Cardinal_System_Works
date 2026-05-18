#include <cardinal/actor/component.hpp>

#include <algorithm>

namespace cardinal::actor {

cardinal::scene::Mat4 TransformComponent::matrix() const {
    using namespace cardinal::scene;
    return Mat4::translation(translation)
         * Mat4::rotation_xyz(rotation_euler)
         * Mat4::scaling(scale);
}

bool TagComponent::has(const std::string& t) const noexcept {
    return std::find(tags.begin(), tags.end(), t) != tags.end();
}

void TagComponent::add(std::string t) {
    if (!has(t)) tags.push_back(std::move(t));
}

void TagComponent::remove(const std::string& t) {
    tags.erase(std::remove(tags.begin(), tags.end(), t), tags.end());
}

}  // namespace cardinal::actor
