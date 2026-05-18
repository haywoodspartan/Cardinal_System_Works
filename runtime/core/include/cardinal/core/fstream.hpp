#pragma once

// =============================================================================
// Cardinal core — file streams (<fstream>) vocabulary (single sanctioned
// include site per the FOUNDATION RULE). Engine code generally prefers the
// higher-level cardinal core/io.hpp; this exposes std file streams for the
// text-format readers/writers (serial, project, …) that want them directly.
// =============================================================================

#include <cardinal/core/types.hpp>   // MASTER typedef header (FOUNDATION RULE)

#include <fstream>
#include <istream>
#include <ostream>
#include <ios>
#include <string>   // std::getline(stream, std::string&) lives here

namespace cardinal {

using std::ifstream;
using std::ofstream;
using std::fstream;
using std::istream;
using std::ostream;
using std::iostream;
using std::ios;
using std::ios_base;
using std::streamsize;
using std::streampos;
using std::streamoff;
using std::getline;
using std::endl;
using std::flush;

}  // namespace cardinal
