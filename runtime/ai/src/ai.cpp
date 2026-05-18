#include <cardinal/ai/ai.hpp>

#include <cardinal/core/algorithm.hpp>   // cardinal::sort/remove_if
#include <cardinal/core/cmath.hpp>       // cardinal::sqrt
// cardinal::get_if/variant/move arrive via ai.hpp (core/utility.hpp)

namespace cardinal::ai {

const char* status_name(Status s) noexcept {
    switch (s) {
        case Status::Running: return "Running";
        case Status::Success: return "Success";
        case Status::Failure: return "Failure";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------
namespace {
template <class T>
T get_or(const cardinal::unordered_map<cardinal::string, BlackboardValue>& m,
         const cardinal::string& k, T def)
{
    auto it = m.find(k);
    if (it == m.end()) return def;
    if (auto* v = cardinal::get_if<T>(&it->second)) return *v;
    return def;
}
}

float Blackboard::get_float(const cardinal::string& k, float v) const            { return get_or(values_, k, v); }
int   Blackboard::get_int  (const cardinal::string& k, int v) const              { return get_or(values_, k, v); }
bool  Blackboard::get_bool (const cardinal::string& k, bool v) const             { return get_or(values_, k, v); }
cardinal::scene::Vec3 Blackboard::get_vec3(const cardinal::string& k, cardinal::scene::Vec3 d) const {
    return get_or(values_, k, d);
}
cardinal::string Blackboard::get_string(const cardinal::string& k, const cardinal::string& d) const {
    return get_or(values_, k, d);
}

cardinal::vector<cardinal::string> Blackboard::keys() const {
    cardinal::vector<cardinal::string> r;
    r.reserve(values_.size());
    for (const auto& [k, _] : values_) r.push_back(k);
    cardinal::sort(r.begin(), r.end());
    return r;
}
const BlackboardValue* Blackboard::value(const cardinal::string& k) const {
    auto it = values_.find(k);
    return it == values_.end() ? nullptr : &it->second;
}

// ---------------------------------------------------------------------------
// Behavior tree composites
// ---------------------------------------------------------------------------
Status Sequence::tick(Blackboard& bb, float dt) {
    while (cursor_ < children_.size()) {
        const Status s = children_[cursor_]->tick(bb, dt);
        if (s == Status::Running) return Status::Running;
        if (s == Status::Failure) { cursor_ = 0; return Status::Failure; }
        ++cursor_;
    }
    cursor_ = 0;
    return Status::Success;
}

Status Selector::tick(Blackboard& bb, float dt) {
    while (cursor_ < children_.size()) {
        const Status s = children_[cursor_]->tick(bb, dt);
        if (s == Status::Running) return Status::Running;
        if (s == Status::Success) { cursor_ = 0; return Status::Success; }
        ++cursor_;
    }
    cursor_ = 0;
    return Status::Failure;
}

Status Inverter::tick(Blackboard& bb, float dt) {
    if (child_ == nullptr) return Status::Failure;
    const Status s = child_->tick(bb, dt);
    if (s == Status::Running) return Status::Running;
    return s == Status::Success ? Status::Failure : Status::Success;
}

Status Wait::tick(Blackboard& /*bb*/, float dt) {
    elapsed_ += dt;
    if (elapsed_ >= duration_) {
        elapsed_ = 0.0f;
        return Status::Success;
    }
    return Status::Running;
}

Status SetBlackboardFloat::tick(Blackboard& bb, float /*dt*/) {
    bb.set_float(key_, value_);
    return Status::Success;
}

// ---------------------------------------------------------------------------
// Perception
// ---------------------------------------------------------------------------
SensorId PerceptionWorld::add_sensor(const SensorDesc& d) {
    SensorEntry e{}; e.id = next_sensor_++; e.d = d; e.alive = true;
    sensors_.push_back(cardinal::move(e));
    return sensors_.back().id;
}
void PerceptionWorld::update_sensor(SensorId id, const SensorDesc& d) {
    for (auto& s : sensors_) if (s.id == id) { s.d = d; return; }
}
void PerceptionWorld::remove_sensor(SensorId id) {
    for (auto& s : sensors_) if (s.id == id) s.alive = false;
}

StimulusId PerceptionWorld::add_stimulus(const StimulusDesc& d) {
    StimulusEntry e{}; e.id = next_stimulus_++; e.d = d; e.age = 0.0f; e.alive = true;
    stimuli_.push_back(cardinal::move(e));
    return stimuli_.back().id;
}
void PerceptionWorld::remove_stimulus(StimulusId id) {
    for (auto& s : stimuli_) if (s.id == id) s.alive = false;
}

void PerceptionWorld::tick(float dt) {
    last_events_.clear();
    // Age stimuli; remove dead.
    for (auto& s : stimuli_) {
        if (!s.alive) continue;
        if (s.d.lifetime_seconds > 0.0f) {
            s.age += dt;
            if (s.age >= s.d.lifetime_seconds) s.alive = false;
        }
    }
    // Compute events.
    for (const auto& sensor : sensors_) {
        if (!sensor.alive) continue;
        for (const auto& stim : stimuli_) {
            if (!stim.alive) continue;
            const auto& sp = sensor.d.position;
            const auto& tp = stim.d.position;
            const f32 dx = tp.x - sp.x, dy = tp.y - sp.y, dz = tp.z - sp.z;
            const f32 dist = cardinal::sqrt(dx*dx + dy*dy + dz*dz);
            if (stim.d.kind == StimulusKind::Sight) {
                if (!sensor.d.sight_enabled) continue;
                if (dist > sensor.d.sight_radius) continue;
                if (dist < 1e-4f) {
                    last_events_.push_back({sensor.id, stim.id, stim.d.kind, tp, 0.0f});
                    continue;
                }
                const cardinal::scene::Vec3 to_target{ dx / dist, dy / dist, dz / dist };
                const cardinal::scene::Vec3 fw = sensor.d.forward;
                const f32 cos_a = to_target.x*fw.x + to_target.y*fw.y + to_target.z*fw.z;
                if (cos_a >= sensor.d.sight_cos_angle) {
                    last_events_.push_back({sensor.id, stim.id, stim.d.kind, tp, dist});
                }
            } else {
                if (!sensor.d.hearing_enabled) continue;
                const f32 effective_radius = sensor.d.hearing_radius * stim.d.strength;
                if (dist <= effective_radius) {
                    last_events_.push_back({sensor.id, stim.id, stim.d.kind, tp, dist});
                }
            }
        }
    }
    // Compact dead.
    sensors_.erase(cardinal::remove_if(sensors_.begin(), sensors_.end(),
        [](const SensorEntry& s){ return !s.alive; }), sensors_.end());
    stimuli_.erase(cardinal::remove_if(stimuli_.begin(), stimuli_.end(),
        [](const StimulusEntry& s){ return !s.alive; }), stimuli_.end());
}

}  // namespace cardinal::ai
