#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Hardware-independent protocol and watchdog, exercised by the native tests.
class Controller {
public:
    float left = 0, right = 0;
    bool estop = false;
    bool limit_reached = false;
    // Consume actual issued half-steps, including turning/reversals. Reserve one
    // step before the next wheel update so quantization never exceeds the budget.
    void wheel_steps(int32_t l, int32_t r, float steps_per_mm) {
        if (limited_) {
            travel_mm_ += (std::fabs(double(l) - previous_left_) +
                           std::fabs(double(r) - previous_right_)) / (2 * steps_per_mm);
            if (travel_mm_ + 1.0 / steps_per_mm >= budget_mm_) {
                limit_reached = true;
                halt();
            }
        }
        previous_left_ = l; previous_right_ = r;
    }
    Controller(bool ready, float limit, float accel, float stop_accel)
        : ready_(ready), limit_(limit), accel_(accel), stop_accel_(stop_accel) {}

    void tick(uint32_t now, bool emergency) {
        const uint32_t dt = now - last_tick_;
        last_tick_ = now;
        estop = emergency;
        if (limit_reached || estop || (driving_ && uint32_t(now - last_drive_) >= 300)) {
            halt();
            return;
        }
        const float step = (stopping_ ? stop_accel_ : accel_) * std::min(dt, uint32_t(50)) / 1000;
        left = approach(left, target_left_, step);
        right = approach(right, target_right_, step);
    }

    // Returns an empty string for successful DRIVE (no reply in section 9.5).
    const char* command(char* line, uint32_t now) {
        char* context = nullptr;
        char* cmd = strtok_r(line, " \t\r", &context);
        char* a = strtok_r(nullptr, " \t\r", &context);
        char* b = strtok_r(nullptr, " \t\r", &context);
        char* extra = strtok_r(nullptr, " \t\r", &context);
        if (!cmd) return "";
        if (!strcmp(cmd, "STOP") && !a) {
            target_left_ = target_right_ = 0;
            stopping_ = true;
            return "OK STOP";
        }
        if (!strcmp(cmd, "LIMIT") && a && !b) {
            float mm;
            if (!number(a, mm) || mm <= 0 || mm > 6096 || left || right || target_left_ || target_right_) {
                halt(); return "ERR invalid travel limit";
            }
            budget_mm_ = mm; travel_mm_ = 0; limited_ = true; limit_reached = false;
            return "OK LIMIT";
        }
        if (!strcmp(cmd, "PING") && !a) return "PONG";
        if (!strcmp(cmd, "LIFT_HOME") || !strcmp(cmd, "LIFT_TO")
                || !strcmp(cmd, "TRAY") || !strcmp(cmd, "TARE")) return "ERR no lift";
        if (!strcmp(cmd, "DRIVE") && a && b && !extra) {
            float l, r;
            if (!number(a, l) || !number(b, r) || fabsf(l) > limit_ || fabsf(r) > limit_) {
                halt();
                return "ERR invalid speed";
            }
            if (limit_reached) { halt(); return "LIMIT REACHED"; }
            if (estop || !ready_) {
                halt();
                return estop ? "ERR estop" : "ERR unconfigured motors";
            }
            if (!limited_) { halt(); return "ERR travel limit required"; }
            target_left_ = l;
            target_right_ = r;
            last_drive_ = now;
            driving_ = true;
            stopping_ = false;
            return "";
        }
        halt();
        return "ERR invalid command";
    }

    void halt() { left = right = target_left_ = target_right_ = 0; driving_ = false; }
private:
    bool limited_ = false;
    double budget_mm_ = 0, travel_mm_ = 0;
    int32_t previous_left_ = 0, previous_right_ = 0;
    bool ready_, driving_ = false, stopping_ = false;
    float limit_, accel_, stop_accel_, target_left_ = 0, target_right_ = 0;
    uint32_t last_drive_ = 0, last_tick_ = 0;
    static float approach(float x, float target, float step) {
        return x + std::max(-step, std::min(step, target - x));
    }
    static bool number(const char* value, float& result) {
        // Plain decimal only (no hex floats, inf, nan): [+-]digits[.digits][e[+-]digits].
        const char* p = value;
        if (*p == '+' || *p == '-') ++p;
        int digits = 0;
        while (*p >= '0' && *p <= '9') { ++p; ++digits; }
        if (*p == '.') { ++p; while (*p >= '0' && *p <= '9') { ++p; ++digits; } }
        if (!digits) return false;
        if (*p == 'e' || *p == 'E') {
            ++p;
            if (*p == '+' || *p == '-') ++p;
            if (*p < '0' || *p > '9') return false;
            while (*p >= '0' && *p <= '9') ++p;
        }
        if (*p) return false;
        result = strtof(value, nullptr);
        return std::isfinite(result);
    }
};
