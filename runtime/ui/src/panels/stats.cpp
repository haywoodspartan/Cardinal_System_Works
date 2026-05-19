// =============================================================================
// Studio — stats panel implementation.
// =============================================================================
#include "stats.hpp"

#include <cardinal/rhi/rhi.hpp>

#include <cardinal/ui/imgui.hpp>

namespace cardinal::ui::panels::stats_panel {

void draw(const char* title, bool* p_open, const Inputs& in) {
    if (!ImGui::Begin(title ? title : "Stats", p_open,
                      ImGuiWindowFlags_NoMove)) { ImGui::End(); return; }

    if (in.device == nullptr) {
        ImGui::TextDisabled("(no device)");
        ImGui::End();
        return;
    }

    const auto& caps = in.device->capabilities();
    const auto& s    = in.device->settings();

    if (ImGui::CollapsingHeader("Adapter", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Name:    %s", in.device->adapter_name());
        ImGui::Text("Vendor:  %s", caps.vendor_name);
        ImGui::Text("Arch:    %s", caps.gpu_arch);
        ImGui::Text("Backend: %s",
            in.device->backend() == cardinal::rhi::Backend::Vulkan ? "Vulkan" : "D3D12");
        ImGui::Text("VRAM:    %llu MB",
            static_cast<unsigned long long>(caps.vram_bytes / (1024ull * 1024ull)));
    }

    if (ImGui::CollapsingHeader("Capabilities", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto row = [](const char* k, bool v) {
            ImGui::TextDisabled("%-26s", k);
            ImGui::SameLine();
            ImGui::TextColored(v ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f)
                                 : ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                               "%s", v ? "yes" : "no");
        };
        row("Dynamic rendering",     caps.dynamic_rendering);
        row("Synchronization2",      caps.synchronization2);
        row("Buffer device address", caps.buffer_device_address);
        row("Descriptor indexing",   caps.descriptor_indexing);
        row("Mesh shaders",          caps.mesh_shader);
        row("Task shaders",          caps.task_shader);
        row("Ray tracing pipeline",  caps.ray_tracing_pipeline);
        row("Ray query",             caps.ray_query);
        row("Acceleration structs",  caps.acceleration_structure);
        row("FP16 (Rapid Math)",     caps.shader_float16);
        row("INT8",                  caps.shader_int8);
        row("Wave intrinsics",       caps.wave_intrinsics);
        row("DLSS capable (NV)",     caps.nvidia_dlss_capable);
        row("Frame Gen capable",     caps.nvidia_framegen_capable);
        row("FSR3 capable (AMD)",    caps.amd_fsr3_capable);
    }

    if (ImGui::CollapsingHeader("Render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* upscale_name =
            s.upscaler == cardinal::rhi::UpscalerMode::Off                   ? "Off (native)" :
            s.upscaler == cardinal::rhi::UpscalerMode::DLSS_Quality          ? "DLSS Quality" :
            s.upscaler == cardinal::rhi::UpscalerMode::DLSS_Balanced         ? "DLSS Balanced" :
            s.upscaler == cardinal::rhi::UpscalerMode::DLSS_Performance      ? "DLSS Performance" :
            s.upscaler == cardinal::rhi::UpscalerMode::DLSS_UltraPerformance ? "DLSS Ultra-Perf" :
            s.upscaler == cardinal::rhi::UpscalerMode::DLAA                  ? "DLAA" :
            s.upscaler == cardinal::rhi::UpscalerMode::FSR3_Quality          ? "FSR3 Quality" :
            s.upscaler == cardinal::rhi::UpscalerMode::FSR3_Balanced         ? "FSR3 Balanced" :
            s.upscaler == cardinal::rhi::UpscalerMode::FSR3_Performance      ? "FSR3 Performance" :
            s.upscaler == cardinal::rhi::UpscalerMode::TAA                   ? "TAA" : "(unknown)";
        ImGui::Text("Upscaler:        %s", upscale_name);
        ImGui::Text("Render scale:    %.2f", s.render_scale);
        ImGui::Text("Frame gen:       %s", s.enable_frame_generation ? "on" : "off");
        ImGui::Text("Mesh shaders:    %s", s.enable_mesh_shaders     ? "on" : "off");
        ImGui::Text("Ray tracing:     %s", s.enable_ray_tracing      ? "on" : "off");
        ImGui::Text("Ray query:       %s", s.enable_ray_query        ? "on" : "off");
        ImGui::Text("Path tracing:    %s%s",
            s.enable_path_tracing ? "on" : "off",
            s.enable_path_tracing ? "" : "");
        if (s.enable_path_tracing) {
            ImGui::Text("    samples/pixel: %u", s.path_tracing_samples_per_pixel);
        }
        ImGui::Text("Prefer FP16:     %s", s.prefer_fp16 ? "yes" : "no");
    }

    if (ImGui::CollapsingHeader("Frame", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("FPS:        %6.1f", in.ema_fps);
        ImGui::Text("Frame time: %6.3f ms",
            in.ema_fps > 0.0f ? 1000.0f / in.ema_fps : 0.0f);
        ImGui::Text("Frames:     %llu",
            static_cast<unsigned long long>(in.frame_count));
        if (in.swapchain != nullptr) {
            ImGui::Text("Swap size:  %u x %u",
                in.swapchain->width(), in.swapchain->height());
            ImGui::Text("Viewport:   %u x %u%s",
                in.swapchain->viewport_width(), in.swapchain->viewport_height(),
                (in.swapchain->viewport_width() == 0) ? " (off)" : "");
        }
    }

    ImGui::End();
}

}  // namespace cardinal::ui::panels::stats_panel
