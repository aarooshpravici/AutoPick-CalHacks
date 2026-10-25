#pragma once
#include <cmath>
#include <cstdint>

// Nonblocking 28BYJ-48 half-step sequencer, ordered ULN2003 IN1..IN4.
class CoilStepper {
public:
    int32_t steps = 0; // Logical wheel direction, independent of wiring inversion.
    uint8_t mask = 0;  // Coils off at boot; STOP/watchdog preserve the last phase.
    bool update(float speed, uint32_t now, float steps_per_mm, bool inverted) {
        if (!initialized_) { last_ = now; initialized_ = true; }
        uint32_t elapsed = now - last_;
        last_ = now;
        if (speed == 0) { phase_fraction_ = 0; return false; }
        // No catch-up burst after scheduler stalls, and no stale fraction on reversal.
        int direction = speed > 0 ? 1 : -1;
        if (direction != last_direction_) phase_fraction_ = 0;
        last_direction_ = direction;
        phase_fraction_ += std::fmin(elapsed, 10000u) * 1e-6 * std::fabs(speed) * steps_per_mm;
        if (phase_fraction_ < 1) return false;
        phase_fraction_ -= std::floor(phase_fraction_);
        phase_ = (phase_ + (inverted ? -direction : direction) + 8) % 8;
        mask = sequence_[phase_];
        steps += direction;
        return true;
    }
private:
    static constexpr uint8_t sequence_[8] = {0x1, 0x3, 0x2, 0x6, 0x4, 0xC, 0x8, 0x9};
    uint32_t last_ = 0;
    bool initialized_ = false;
    int phase_ = 0, last_direction_ = 0;
    double phase_fraction_ = 0;
};
