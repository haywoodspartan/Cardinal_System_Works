// =============================================================================
// Studio — Shader compiler panel implementation.
// =============================================================================
#include "shader_compiler.hpp"

#include <cardinal/shader/shader.hpp>

#include <cardinal/ui/imgui.hpp>

#include <cardinal/core/cstdio.hpp>

namespace cardinal::ui::panels::shader_compiler_panel {

void draw(cardinal::shader::Compiler* compiler, const char* title, bool* p_open) {
    if (!ImGui::Begin(title ? title : "Shader Compiler", p_open,
                      0)) { ImGui::End(); return; }
    if (compiler == nullptr) {
        ImGui::TextDisabled("(no shader::Compiler bound)");
        ImGui::End();
        return;
    }

    static char        source[16 * 1024] =
        "// HLSL shader — try editing me!\n"
        "struct VSIn  { float3 pos : POSITION; float3 col : COLOR; };\n"
        "struct VSOut { float4 pos : SV_POSITION; float3 col : COLOR; };\n"
        "VSOut VSMain(VSIn v) { VSOut o; o.pos = float4(v.pos, 1.0); o.col = v.col; return o; }\n"
        "float4 PSMain(VSOut i) : SV_TARGET { return float4(i.col, 1.0); }\n";
    static char        entry[64]   = "VSMain";
    static int         stage_idx   = 0;
    static char        defines[256]= "";
    static cardinal::shader::CompileResult last;
    static bool        ever_compiled = false;

    const auto stats = compiler->stats();
    ImGui::Text("Cache:        %u blobs (%.1f KiB) | hits %llu / compiles %llu",
        stats.cached_blobs,
        static_cast<double>(stats.cache_bytes) / 1024.0,
        static_cast<unsigned long long>(stats.cache_hits),
        static_cast<unsigned long long>(stats.compiles_total));
    ImGui::Text("Watched:      %u files | hot reloads: %llu",
        stats.watched_files,
        static_cast<unsigned long long>(stats.hot_reloads));

    ImGui::Separator();

    const char* stages[] = { "Vertex", "Fragment", "Compute" };
    ImGui::Combo("Stage", &stage_idx, stages, IM_ARRAYSIZE(stages));
    ImGui::InputText("Entry", entry, sizeof(entry));
    ImGui::InputText("Defines (space-sep)", defines, sizeof(defines));
    ImGui::TextDisabled("Source HLSL:");
    ImGui::InputTextMultiline("##src", source, sizeof(source),
        ImVec2(-FLT_MIN, 240.0f), ImGuiInputTextFlags_AllowTabInput);

    if (ImGui::Button("Compile", ImVec2(140.0f, 0.0f))) {
        cardinal::shader::CompileRequest r{};
        r.source_text = source;
        r.entry_point = entry;
        r.stage       = static_cast<cardinal::shader::Stage>(stage_idx);
        // Tokenise defines on whitespace.
        cardinal::string tok;
        for (const char* p = defines; *p != 0; ++p) {
            if (*p == ' ' || *p == '\t' || *p == ',') {
                if (!tok.empty()) { r.defines.push_back(tok); tok.clear(); }
            } else tok.push_back(*p);
        }
        if (!tok.empty()) r.defines.push_back(tok);
        last = compiler->compile(r);
        ever_compiled = true;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(cached results return instantly)");

    if (ever_compiled) {
        ImGui::Separator();
        if (last.ok) {
            ImGui::TextColored(ImVec4(0.4f, 0.95f, 0.4f, 1.0f),
                "OK — %zu bytes (%s, %.3fms)",
                last.bytecode.size(),
                last.served_from_cache ? "cached" : "compiled",
                last.compile_seconds * 1000.0);
        } else {
            ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "FAIL");
        }
        if (!last.diagnostics.empty()) {
            ImGui::TextDisabled("diagnostics:");
            ImGui::TextWrapped("%s", last.diagnostics.c_str());
        }
        ImGui::TextDisabled("cache key: %016llx",
            static_cast<unsigned long long>(last.cache_key));
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::shader_compiler_panel
