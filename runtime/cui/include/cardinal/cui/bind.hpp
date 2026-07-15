#pragma once

// =============================================================================
// Cardinal UI — templated value-binding widgets (core building blocks).
//
// The base widget set is deliberately type-free; this header layers TEMPLATED
// bindings on top so panels can wire engine values with one line:
//
//     stack->add(cardinal::make_unique<NumberField<float>>(&exposure, 0.f, 8.f));
//     stack->add(cardinal::make_unique<EnumCombo<scene::ViewMode>>(
//         &mode, labels));
//
// The bindings ride the PUBLIC widget virtuals (on_key / on_focus_changed /
// on_click), NOT the on_submit/on_change/on_select callback slots — those are
// single-slot cardinal::functions and stay free for the panel's own hooks.
//
// Ownership contract: the bound pointer must outlive the widget (same rule as
// Checkbox's bool*). refresh() pulls value -> widget when external code
// mutates the value.
//
// NOTE (EnumCombo): it inherits Combo's host popup contract — while open,
// hand it to Ui::set_popup so the dropdown wins input over later siblings.
// =============================================================================

#include <cardinal/core/std/cstdio.hpp>    // cardinal::snprintf
#include <cardinal/core/std/cstdlib.hpp>   // cardinal::strtod

#include "widgets.hpp"

namespace cardinal::cui {

// -----------------------------------------------------------------------------
// NumberField<T> — a TextField bound to a numeric value (int/u32/float/double).
// Commits on Enter AND on focus loss (click-away / Escape); live=true also
// adopts every keystroke. The parsed value is clamped to [min,max]; unparsable
// text reverts; a commit canonicalises the text (e.g. "007" -> "7").
// -----------------------------------------------------------------------------
template <class T>
class NumberField : public TextField {
public:
    NumberField(T* value, T min_v, T max_v,
                bool live = false, float min_width = 120.0f)
        : TextField({}, min_width),
          value_(value), min_(min_v), max_(max_v), live_(live) {
        refresh();
    }

    // Pull the bound value into the text (external mutation happened).
    void refresh() {
        if (value_ == nullptr) return;
        char b[48];
        format_(b, sizeof(b), *value_);
        set_text(cardinal::string(b));
        set_caret(text().size());
    }

    T clamped(double v) const noexcept {
        if (v < static_cast<double>(min_)) v = static_cast<double>(min_);
        if (v > static_cast<double>(max_)) v = static_cast<double>(max_);
        return static_cast<T>(v);
    }

    void on_key(const InputState& in) override {
        TextField::on_key(in);                 // normal editing (+ user on_submit)
        if (in.key_enter)  adopt_(true);       // commit + canonicalise
        else if (live_)    adopt_(false);      // track every edit
    }

    void on_focus_changed(bool focused) override {
        TextField::on_focus_changed(focused);
        if (!focused) adopt_(true);            // commit on blur (click-away/Esc)
    }

private:
    static void format_(char* b, usize n, float v) {
        cardinal::snprintf(b, n, "%.6g", static_cast<double>(v));
    }
    static void format_(char* b, usize n, double v) { cardinal::snprintf(b, n, "%.9g", v); }
    static void format_(char* b, usize n, i32 v)    { cardinal::snprintf(b, n, "%d", v); }
    static void format_(char* b, usize n, u32 v)    { cardinal::snprintf(b, n, "%u", v); }
    static void format_(char* b, usize n, i64 v) {
        cardinal::snprintf(b, n, "%lld", static_cast<long long>(v));
    }
    static void format_(char* b, usize n, u64 v) {
        cardinal::snprintf(b, n, "%llu", static_cast<unsigned long long>(v));
    }

    void adopt_(bool canonicalise) {
        if (value_ == nullptr) return;
        const char* s   = text().c_str();
        char*       end = nullptr;
        const double v = cardinal::strtod(s, &end);
        if (end == s) {              // unparsable -> revert to the bound value
            if (canonicalise) refresh();
            return;
        }
        *value_ = clamped(v);
        if (canonicalise) refresh(); // commit shows the clamped, canonical form
    }

    T*   value_ {nullptr};
    T    min_ {};
    T    max_ {};
    bool live_ {false};
};

// -----------------------------------------------------------------------------
// EnumCombo<E> — a Combo bound to an enum (or any int-castable) value.
// Two forms:
//   EnumCombo(&v, labels)          — enum values are 0..N-1 (label order)
//   EnumCombo(&v, labels, values)  — explicit value table (non-contiguous
//                                    enums; values[i] pairs with labels[i])
// -----------------------------------------------------------------------------
template <class E>
class EnumCombo : public Combo {
public:
    EnumCombo(E* value, cardinal::vector<cardinal::string> labels)
        : value_(value) {
        for (usize i = 0; i < labels.size(); ++i)
            values_.push_back(static_cast<E>(static_cast<int>(i)));
        set_items(cardinal::move(labels));
        refresh();
    }
    EnumCombo(E* value, cardinal::vector<cardinal::string> labels,
              cardinal::vector<E> values)
        : value_(value), values_(cardinal::move(values)) {
        set_items(cardinal::move(labels));
        refresh();
    }

    void on_click() override {
        Combo::on_click();                     // selection resolved here
        const int i = selected();
        if (value_ != nullptr && i >= 0 && i < static_cast<int>(values_.size()))
            *value_ = values_[static_cast<usize>(i)];
    }

    // Pull the bound value into the selection (external mutation happened).
    void refresh() {
        if (value_ == nullptr) return;
        for (usize i = 0; i < values_.size(); ++i) {
            if (values_[i] == *value_) {
                set_selected(static_cast<int>(i));
                return;
            }
        }
        set_selected(-1);
    }

private:
    E*                  value_ {nullptr};
    cardinal::vector<E> values_;
};

}  // namespace cardinal::cui
