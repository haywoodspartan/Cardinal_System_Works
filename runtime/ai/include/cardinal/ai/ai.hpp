#pragma once

// =============================================================================
// Cardinal — AI Core.
//
// Three pieces, designed to compose:
//
//   Blackboard — typed key/value store. Behavior trees + perception both
//                read/write here. Keys are strings; values are float / int /
//                bool / vec3 / string. Queries return a default if missing.
//
//   Behavior Tree — composite + leaf nodes. Standard set: Sequence,
//                Selector, Parallel, Inverter, Repeater (composites),
//                Wait, MoveTo, SetBlackboard, Custom (leaves). Tick returns
//                Running / Success / Failure.
//
//   Perception — sight (cone) + hearing (radius) sensors registered to
//                an AI agent. PerceptionWorld owns "stimuli" (sound source
//                positions, sight-targets) + per-frame computes which agent
//                sees / hears what. Agents subscribe via callbacks.
//
// All three live on top of cardinal::geom for ray + cone math. AI agents
// can be either actor::Actor instances (one BT per actor) or mass::Entity
// rows (one BT per entity, much higher density).
// =============================================================================

#include <cardinal/core/types.hpp>        // function/string/unique_ptr
#include <cardinal/core/utility.hpp>      // cardinal::variant, cardinal::move
#include <cardinal/core/containers.hpp>   // unordered_map, vector
#include <cardinal/scene/math.hpp>

namespace cardinal::ai {

// ---------------------------------------------------------------------------
// Blackboard
// ---------------------------------------------------------------------------
using BlackboardValue = cardinal::variant<float, int, bool, cardinal::scene::Vec3, cardinal::string>;

class Blackboard {
public:
    void set_float (const cardinal::string& k, float v)                       { values_[k] = v; }
    void set_int   (const cardinal::string& k, int v)                         { values_[k] = v; }
    void set_bool  (const cardinal::string& k, bool v)                        { values_[k] = v; }
    void set_vec3  (const cardinal::string& k, const cardinal::scene::Vec3& v){ values_[k] = v; }
    void set_string(const cardinal::string& k, const cardinal::string& v)          { values_[k] = v; }

    float                 get_float (const cardinal::string& k, float v_default = 0.0f) const;
    int                   get_int   (const cardinal::string& k, int v_default = 0) const;
    bool                  get_bool  (const cardinal::string& k, bool v_default = false) const;
    cardinal::scene::Vec3 get_vec3  (const cardinal::string& k, cardinal::scene::Vec3 d = {0,0,0}) const;
    cardinal::string           get_string(const cardinal::string& k, const cardinal::string& d = {}) const;

    bool   has(const cardinal::string& k) const { return values_.find(k) != values_.end(); }
    void   clear() { values_.clear(); }

    // Inspection — for the AI panel.
    cardinal::vector<cardinal::string> keys() const;
    const BlackboardValue*   value(const cardinal::string& k) const;

private:
    cardinal::unordered_map<cardinal::string, BlackboardValue> values_;
};

// ---------------------------------------------------------------------------
// Behavior tree
// ---------------------------------------------------------------------------
enum class Status : u32 { Running, Success, Failure };
const char* status_name(Status s) noexcept;

class Node {
public:
    virtual ~Node() = default;
    virtual Status     tick(Blackboard& bb, float dt) = 0;
    virtual const char* name() const noexcept = 0;
    virtual u32        child_count() const noexcept { return 0; }
    virtual Node*      child(u32 /*i*/) const noexcept { return nullptr; }
};

class Sequence : public Node {
public:
    Sequence& add(cardinal::unique_ptr<Node> child) { children_.push_back(cardinal::move(child)); return *this; }
    Status tick(Blackboard& bb, float dt) override;
    const char* name() const noexcept override { return "Sequence"; }
    u32   child_count() const noexcept override { return static_cast<u32>(children_.size()); }
    Node* child(u32 i) const noexcept override { return i < children_.size() ? children_[i].get() : nullptr; }
private:
    cardinal::vector<cardinal::unique_ptr<Node>> children_;
    u32 cursor_{0};
};

class Selector : public Node {
public:
    Selector& add(cardinal::unique_ptr<Node> child) { children_.push_back(cardinal::move(child)); return *this; }
    Status tick(Blackboard& bb, float dt) override;
    const char* name() const noexcept override { return "Selector"; }
    u32   child_count() const noexcept override { return static_cast<u32>(children_.size()); }
    Node* child(u32 i) const noexcept override { return i < children_.size() ? children_[i].get() : nullptr; }
private:
    cardinal::vector<cardinal::unique_ptr<Node>> children_;
    u32 cursor_{0};
};

class Inverter : public Node {
public:
    explicit Inverter(cardinal::unique_ptr<Node> c) : child_(cardinal::move(c)) {}
    Status tick(Blackboard& bb, float dt) override;
    const char* name() const noexcept override { return "Inverter"; }
    u32   child_count() const noexcept override { return child_ ? 1u : 0u; }
    Node* child(u32 i) const noexcept override { return (i == 0) ? child_.get() : nullptr; }
private:
    cardinal::unique_ptr<Node> child_;
};

class Wait : public Node {
public:
    explicit Wait(float seconds) : duration_(seconds) {}
    Status tick(Blackboard& bb, float dt) override;
    const char* name() const noexcept override { return "Wait"; }
private:
    float duration_{1.0f};
    float elapsed_{0.0f};
};

class SetBlackboardFloat : public Node {
public:
    SetBlackboardFloat(cardinal::string k, float v) : key_(cardinal::move(k)), value_(v) {}
    Status tick(Blackboard& bb, float dt) override;
    const char* name() const noexcept override { return "SetBB"; }
private:
    cardinal::string key_;
    float       value_;
};

// Custom — host-supplied lambda. Convenience for "everything else".
class Custom : public Node {
public:
    using Fn = cardinal::function<Status(Blackboard&, float)>;
    Custom(cardinal::string n, Fn fn) : name_(cardinal::move(n)), fn_(cardinal::move(fn)) {}
    Status tick(Blackboard& bb, float dt) override { return fn_ ? fn_(bb, dt) : Status::Failure; }
    const char* name() const noexcept override { return name_.c_str(); }
private:
    cardinal::string name_;
    Fn          fn_;
};

// ---------------------------------------------------------------------------
// Perception
// ---------------------------------------------------------------------------
using SensorId   = u32;
using StimulusId = u32;

struct SensorDesc {
    cardinal::scene::Vec3 position{0, 1.7f, 0};   // eye / ear position
    cardinal::scene::Vec3 forward {0, 0, -1};     // forward (sight cone axis)
    f32  sight_radius   {25.0f};
    f32  sight_cos_angle{0.5f};                   // cos(half-FOV) — 0.5 ≈ 60°
    f32  hearing_radius {15.0f};
    bool sight_enabled  {true};
    bool hearing_enabled{true};
};

enum class StimulusKind : u32 { Sight = 0, Sound = 1 };
struct StimulusDesc {
    StimulusKind        kind{StimulusKind::Sight};
    cardinal::scene::Vec3 position{0,0,0};
    f32                 strength{1.0f};         // sound: louder = farther
    f32                 lifetime_seconds{0.5f}; // 0 = never expires
    u32                 source_tag{0};
};

struct PerceptionEvent {
    SensorId    sensor;
    StimulusId  stimulus;
    StimulusKind kind;
    cardinal::scene::Vec3 position;
    f32          distance;
};

class PerceptionWorld {
public:
    SensorId   add_sensor   (const SensorDesc& d);
    void       update_sensor(SensorId id, const SensorDesc& d);
    void       remove_sensor(SensorId id);

    StimulusId add_stimulus (const StimulusDesc& d);
    void       remove_stimulus(StimulusId id);

    // Tick: ages out timed stimuli + computes events for this frame. Each
    // sensor that perceives a stimulus produces an event. Cleared at the
    // start of each tick.
    void tick(float dt);
    const cardinal::vector<PerceptionEvent>& last_events() const noexcept { return last_events_; }

    usize sensor_count()   const noexcept { return sensors_.size(); }
    usize stimulus_count() const noexcept { return stimuli_.size(); }

private:
    struct SensorEntry   { SensorId   id; SensorDesc d; bool alive{true}; };
    struct StimulusEntry { StimulusId id; StimulusDesc d; f32 age{0.0f}; bool alive{true}; };
    cardinal::vector<SensorEntry>          sensors_;
    cardinal::vector<StimulusEntry>        stimuli_;
    cardinal::vector<PerceptionEvent>      last_events_;
    SensorId                          next_sensor_{1};
    StimulusId                        next_stimulus_{1};
};

}  // namespace cardinal::ai
