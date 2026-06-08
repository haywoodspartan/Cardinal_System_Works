// =============================================================================
// Cardinal UI — Cardinal Slate (cui) -> ImGui draw-list bridge implementation.
// =============================================================================
#include <cardinal/ui/cui_render.hpp>

#include <imgui.h>

namespace cardinal::ui {

namespace {
inline ImU32 to_col(cardinal::cui::Color c) noexcept {
    return IM_COL32(c.r, c.g, c.b, c.a);
}
inline ImVec2 to_pt(cardinal::cui::Vec2 v, cardinal::cui::Vec2 o) noexcept {
    return ImVec2(v.x + o.x, v.y + o.y);
}
}  // namespace

void cui_render(const cardinal::cui::DrawList& dl,
                void* imgui_draw_list,
                cardinal::cui::Vec2 origin) noexcept {
    auto* dr = static_cast<ImDrawList*>(imgui_draw_list);
    if (dr == nullptr) return;

    namespace cui = cardinal::cui;
    for (const auto& cmd : dl.cmds()) {
        const ImVec2 a = to_pt(cmd.rect.pos, origin);
        switch (cmd.kind) {
        case cui::DrawKind::RectFilled: {
            const ImVec2 b(a.x + cmd.rect.size.x, a.y + cmd.rect.size.y);
            dr->AddRectFilled(a, b, to_col(cmd.color), cmd.rounding);
            break;
        }
        case cui::DrawKind::RectStroke: {
            const ImVec2 b(a.x + cmd.rect.size.x, a.y + cmd.rect.size.y);
            dr->AddRect(a, b, to_col(cmd.color), cmd.rounding, 0, cmd.thickness);
            break;
        }
        case cui::DrawKind::Line: {
            dr->AddLine(a, to_pt(cmd.p1, origin), to_col(cmd.color), cmd.thickness);
            break;
        }
        case cui::DrawKind::Text: {
            const float fs = cmd.font_size > 0.0f ? cmd.font_size : ImGui::GetFontSize();
            dr->AddText(ImGui::GetFont(), fs, a, to_col(cmd.color), cmd.text.c_str());
            break;
        }
        }
    }
}

}  // namespace cardinal::ui
