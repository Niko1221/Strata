// Runtime VRAM floor. Arithmetic stays in vram_cap.hpp (no device calls). This file is the
// device check used after allocations the startup reading cannot see. Disarmed (cap off):
// every function returns without a device call, so the default path is unchanged.
#pragma once

#include <string>

namespace strata::core {

void vram_floor_arm(double fraction);
bool vram_floor_armed();

// False when the cap is on and this device's free bytes are under the floor. err is set.
// The refusal line matches check_vram_cap ("cap not relaxed").
bool vram_floor_allow(const char* where, std::string& err);

// One stderr line per label: "strata vram: <label> free_mib=N". No device call when disarmed.
void vram_floor_log_once(const char* label);

}  // namespace strata::core
