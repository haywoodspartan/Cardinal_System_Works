#pragma once

// =============================================================================
// Cardinal core — <filesystem> vocabulary (single sanctioned include site
// per the FOUNDATION RULE). Non-core code uses cardinal::fs::path etc.;
// core/io.hpp remains the higher-level engine I/O surface.
// =============================================================================

#include <filesystem>

namespace cardinal {

namespace fs = std::filesystem;

}  // namespace cardinal
