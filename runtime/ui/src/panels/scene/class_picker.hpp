#pragma once

// =============================================================================
// Studio — Game Class picker / inspector.
//
// Lists every class registered with CARDINAL_REGISTER_GAME_CLASS, grouped by
// the "Category/" prefix string. Click "Spawn" to instantiate a fresh actor
// of that class (delegates to game::Game::spawn_class). Selected actor's
// reflected properties appear in the lower pane with typed editors.
// =============================================================================

#include <cardinal/core/types.hpp>

namespace cardinal::game { class Game; }

namespace cardinal::ui::panels::class_picker_panel {

void draw(cardinal::game::Game* game,
          cardinal::u32* selected_actor_id_inout,
          const char* title = "Game Classes",
          bool* p_open = nullptr);

}  // namespace cardinal::ui::panels::class_picker_panel
