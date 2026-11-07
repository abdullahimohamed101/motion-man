// SHA-256 (FIPS 180-4) for artifact identity: engines and models are refused unless their bytes match the manifest
// (spec C§5). Output uses the manifests' "sha256:<hex>" form.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace motionman {

std::string sha256_hex(const std::uint8_t* data, std::size_t size);  // "sha256:<64 hex>"
std::string sha256_file(const std::string& path);                     // throws std::runtime_error if unreadable

}  // namespace motionman
