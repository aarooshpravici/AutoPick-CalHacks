#pragma once
// UNCONFIRMED: HW1 must select the board, motor backend and every pin.
// This provisional backend is STEP/DIR only. Do not use with DC/H-bridge motors.
namespace wiring {
constexpr bool confirmed = false;
constexpr int left_step = -1, left_dir = -1, left_enable = -1;
constexpr int right_step = -1, right_dir = -1, right_enable = -1;
constexpr int estop_sense = -1; // Dedicated isolated/contact input, NEVER connect 12 V.
constexpr bool enable_active_low = true, estop_active_low = true;
constexpr bool invert_left = false, invert_right = false;
constexpr float wheel_diameter_mm = 72.0f; // Replace with measurements.
constexpr int microsteps_per_turn = 1600; // Match physical driver switches.
constexpr float max_speed_mm_s = 100.0f; // First real runs <=100.
constexpr float acceleration_mm_s2 = 200.0f;
constexpr float stop_acceleration_mm_s2 = 2000.0f;
constexpr bool pins_valid = left_step >= 0 && left_dir >= 0 && left_enable >= 0
    && right_step >= 0 && right_dir >= 0 && right_enable >= 0 && estop_sense >= 0;
static_assert(!confirmed || pins_valid, "Confirm every pin before enabling outputs");
}
