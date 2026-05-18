#include <cardinal/input/input.hpp>

#include <algorithm>
#include <cstring>
#include <mutex>

namespace cardinal::input {

const char* key_code_name(KeyCode k) noexcept {
    switch (k) {
        case KeyCode::A: return "A"; case KeyCode::B: return "B"; case KeyCode::C: return "C";
        case KeyCode::D: return "D"; case KeyCode::E: return "E"; case KeyCode::F: return "F";
        case KeyCode::G: return "G"; case KeyCode::H: return "H"; case KeyCode::I: return "I";
        case KeyCode::J: return "J"; case KeyCode::K: return "K"; case KeyCode::L: return "L";
        case KeyCode::M: return "M"; case KeyCode::N: return "N"; case KeyCode::O: return "O";
        case KeyCode::P: return "P"; case KeyCode::Q: return "Q"; case KeyCode::R: return "R";
        case KeyCode::S: return "S"; case KeyCode::T: return "T"; case KeyCode::U: return "U";
        case KeyCode::V: return "V"; case KeyCode::W: return "W"; case KeyCode::X: return "X";
        case KeyCode::Y: return "Y"; case KeyCode::Z: return "Z";
        case KeyCode::K0: return "0"; case KeyCode::K1: return "1"; case KeyCode::K2: return "2";
        case KeyCode::K3: return "3"; case KeyCode::K4: return "4"; case KeyCode::K5: return "5";
        case KeyCode::K6: return "6"; case KeyCode::K7: return "7"; case KeyCode::K8: return "8";
        case KeyCode::K9: return "9";
        case KeyCode::Space:     return "Space";
        case KeyCode::Tab:       return "Tab";
        case KeyCode::Enter:     return "Enter";
        case KeyCode::Escape:    return "Escape";
        case KeyCode::Backspace: return "Backspace";
        case KeyCode::LeftShift: return "LShift"; case KeyCode::RightShift: return "RShift";
        case KeyCode::LeftCtrl:  return "LCtrl";  case KeyCode::RightCtrl:  return "RCtrl";
        case KeyCode::LeftAlt:   return "LAlt";   case KeyCode::RightAlt:   return "RAlt";
        case KeyCode::Up: return "Up"; case KeyCode::Down: return "Down";
        case KeyCode::Left: return "Left"; case KeyCode::Right: return "Right";
        case KeyCode::F1: return "F1"; case KeyCode::F2: return "F2"; case KeyCode::F3: return "F3";
        case KeyCode::F4: return "F4"; case KeyCode::F5: return "F5"; case KeyCode::F6: return "F6";
        case KeyCode::F7: return "F7"; case KeyCode::F8: return "F8"; case KeyCode::F9: return "F9";
        case KeyCode::F10: return "F10"; case KeyCode::F11: return "F11"; case KeyCode::F12: return "F12";
        default: return "Unknown";
    }
}

KeyCode key_code_from_string(const char* s) noexcept {
    if (s == nullptr) return KeyCode::Unknown;
    // Single-character lookup for letters / digits.
    if (s[0] != 0 && s[1] == 0) {
        const char c = s[0];
        if (c >= 'a' && c <= 'z') return static_cast<KeyCode>(static_cast<u32>(KeyCode::A) + (c - 'a'));
        if (c >= 'A' && c <= 'Z') return static_cast<KeyCode>(static_cast<u32>(KeyCode::A) + (c - 'A'));
        if (c >= '0' && c <= '9') return static_cast<KeyCode>(static_cast<u32>(KeyCode::K0) + (c - '0'));
    }
    for (u32 i = 1; i < static_cast<u32>(KeyCode::Count); ++i) {
        const auto k = static_cast<KeyCode>(i);
        if (std::strcmp(s, key_code_name(k)) == 0) return k;
    }
    return KeyCode::Unknown;
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct Manager::Impl {
    std::array<ButtonState, static_cast<usize>(KeyCode::Count)>      keys{};
    MouseState                                                       mouse{};
    std::unordered_map<std::string, Action>                          actions;
    std::unordered_map<std::string, AxisBinding>                     axes;
    std::vector<Binding>                                             empty_bindings;
    std::mutex                                                       mtx;        // OS hook is multi-threaded on some backends
    void*                                                            window_ptr{nullptr};
    u64                                                              events_processed{0};

    // Pending input from the OS hook — drained by begin_frame.
    struct PendKey   { KeyCode k; bool down; };
    struct PendMouse { MouseButton b; bool down; };
    std::vector<PendKey>   pend_keys;
    std::vector<PendMouse> pend_mouse;
    int  pend_mouse_x{-1}, pend_mouse_y{-1};
    int  pend_mouse_wheel{0};

    // PIE gate — when false, action_*() and axis() return false / 0 even
    // if the underlying keys are physically held. Studio drives this from
    // the Game lifecycle (true only while the game is in Playing state).
    // Defaults to true so headless callers and tests get the prior
    // always-on behaviour without an opt-in.
    bool gameplay_active{true};
};

std::shared_ptr<Manager> Manager::create() {
    return std::shared_ptr<Manager>(new Manager());
}

Manager::Manager() : impl_(new Impl{}) {}
Manager::~Manager() { delete impl_; }

void Manager::attach_to_window(void* window_ptr) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->window_ptr = window_ptr;
}
void Manager::detach_from_window() noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->window_ptr = nullptr;
}

void Manager::begin_frame(float dt) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);

    // Pre-pass: copy current "down" → "was_down" via the held_time logic.
    for (auto& k : impl_->keys) {
        const bool was_down = k.down;
        k.pressed  = false;
        k.released = false;
        if (was_down) k.held_time += dt;
    }
    for (auto& b : impl_->mouse.buttons) {
        const bool was_down = b.down;
        b.pressed  = false;
        b.released = false;
        if (was_down) b.held_time += dt;
    }

    // Drain pending key events.
    for (const auto& e : impl_->pend_keys) {
        if (e.k == KeyCode::Unknown || e.k >= KeyCode::Count) continue;
        auto& s = impl_->keys[static_cast<usize>(e.k)];
        if (e.down) {
            if (!s.down) { s.pressed = true; s.held_time = 0.0f; }
            s.down = true;
        } else {
            if (s.down) s.released = true;
            s.down = false;
        }
        ++impl_->events_processed;
    }
    impl_->pend_keys.clear();

    for (const auto& e : impl_->pend_mouse) {
        if (static_cast<u32>(e.b) >= static_cast<u32>(MouseButton::Count)) continue;
        auto& s = impl_->mouse.buttons[static_cast<usize>(e.b)];
        if (e.down) {
            if (!s.down) { s.pressed = true; s.held_time = 0.0f; }
            s.down = true;
        } else {
            if (s.down) s.released = true;
            s.down = false;
        }
        ++impl_->events_processed;
    }
    impl_->pend_mouse.clear();

    // Mouse position + delta.
    if (impl_->pend_mouse_x >= 0) {
        impl_->mouse.dx = impl_->pend_mouse_x - impl_->mouse.x;
        impl_->mouse.dy = impl_->pend_mouse_y - impl_->mouse.y;
        impl_->mouse.x  = impl_->pend_mouse_x;
        impl_->mouse.y  = impl_->pend_mouse_y;
        impl_->pend_mouse_x = -1;
    } else {
        impl_->mouse.dx = 0;
        impl_->mouse.dy = 0;
    }
    impl_->mouse.wheel    = impl_->pend_mouse_wheel;
    impl_->pend_mouse_wheel = 0;
}

const ButtonState& Manager::key(KeyCode k) const noexcept {
    static const ButtonState empty{};
    if (k == KeyCode::Unknown || k >= KeyCode::Count) return empty;
    return impl_->keys[static_cast<usize>(k)];
}
const ButtonState& Manager::mouse(MouseButton b) const noexcept {
    static const ButtonState empty{};
    if (static_cast<u32>(b) >= static_cast<u32>(MouseButton::Count)) return empty;
    return impl_->mouse.buttons[static_cast<usize>(b)];
}
const MouseState& Manager::mouse_state() const noexcept {
    return impl_->mouse;
}

void Manager::bind_action(const std::string& name, Binding b) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->actions[name].name = name;
    impl_->actions[name].bindings.push_back(b);
}
void Manager::bind_axis(const std::string& name, Binding b) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->axes[name].name = name;
    impl_->axes[name].contributions.push_back(b);
}
void Manager::clear_action(const std::string& name) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->actions.erase(name);
}
void Manager::clear_axis(const std::string& name) {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->axes.erase(name);
}
void Manager::reset_bindings() noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->actions.clear();
    impl_->axes.clear();
}

namespace {
const ButtonState& binding_state(const Manager& m, const Binding& b) noexcept {
    static const ButtonState empty{};
    switch (b.kind) {
        case BindKind::Key:           return m.key   (static_cast<KeyCode>(b.code));
        case BindKind::MouseButton:   return m.mouse (static_cast<MouseButton>(b.code));
        case BindKind::GamepadButton: return empty;   // gamepad not yet wired
        case BindKind::GamepadAxis:   return empty;
    }
    return empty;
}
}

bool Manager::action_down(const std::string& name) const {
    if (!impl_->gameplay_active) return false;
    auto it = impl_->actions.find(name);
    if (it == impl_->actions.end()) return false;
    for (const auto& b : it->second.bindings)
        if (binding_state(*this, b).down) return true;
    return false;
}
bool Manager::action_pressed(const std::string& name) const {
    if (!impl_->gameplay_active) return false;
    auto it = impl_->actions.find(name);
    if (it == impl_->actions.end()) return false;
    for (const auto& b : it->second.bindings)
        if (binding_state(*this, b).pressed) return true;
    return false;
}
bool Manager::action_released(const std::string& name) const {
    if (!impl_->gameplay_active) return false;
    auto it = impl_->actions.find(name);
    if (it == impl_->actions.end()) return false;
    for (const auto& b : it->second.bindings)
        if (binding_state(*this, b).released) return true;
    return false;
}

float Manager::axis(const std::string& name) const {
    if (!impl_->gameplay_active) return 0.0f;
    auto it = impl_->axes.find(name);
    if (it == impl_->axes.end()) return 0.0f;
    float sum = 0.0f;
    for (const auto& b : it->second.contributions) {
        const auto& s = binding_state(*this, b);
        if (s.down) sum += b.scale;
    }
    if (sum < -1.0f) sum = -1.0f;
    if (sum >  1.0f) sum =  1.0f;
    return sum;
}

void Manager::set_gameplay_active(bool active) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->gameplay_active = active;
}
bool Manager::is_gameplay_active() const noexcept {
    return impl_->gameplay_active;
}

std::vector<std::string> Manager::action_names() const {
    std::vector<std::string> r;
    for (const auto& [n, _] : impl_->actions) r.push_back(n);
    std::sort(r.begin(), r.end());
    return r;
}
std::vector<std::string> Manager::axis_names() const {
    std::vector<std::string> r;
    for (const auto& [n, _] : impl_->axes) r.push_back(n);
    std::sort(r.begin(), r.end());
    return r;
}
const std::vector<Binding>& Manager::action_bindings(const std::string& name) const {
    auto it = impl_->actions.find(name);
    return it == impl_->actions.end() ? impl_->empty_bindings : it->second.bindings;
}
const std::vector<Binding>& Manager::axis_bindings(const std::string& name) const {
    auto it = impl_->axes.find(name);
    return it == impl_->axes.end() ? impl_->empty_bindings : it->second.contributions;
}

void Manager::push_key(KeyCode k, bool down) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->pend_keys.push_back({k, down});
}
void Manager::push_mouse_button(MouseButton b, bool down) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->pend_mouse.push_back({b, down});
}
void Manager::push_mouse_move(int x, int y) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->pend_mouse_x = x; impl_->pend_mouse_y = y;
}
void Manager::push_mouse_wheel(int delta) noexcept {
    std::lock_guard<std::mutex> lg(impl_->mtx);
    impl_->pend_mouse_wheel += delta;
}

Manager::Stats Manager::stats() const noexcept {
    Stats s{};
    s.events_processed = impl_->events_processed;
    for (const auto& k : impl_->keys) if (k.down) ++s.keys_down_now;
    for (const auto& b : impl_->mouse.buttons) if (b.down) ++s.mouse_buttons_down;
    return s;
}

}  // namespace cardinal::input
