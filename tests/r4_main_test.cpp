#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include "Arduino.h"
#define private public // The fuzz inspects controller internals (budget, last DRIVE).
#include "../base/firmware/src/main.cpp"
#undef private

void advance(unsigned ms) {
    // Wall-clock based: synchronous UART writes also consume test time.
    const uint64_t end = test_time_us + uint64_t(ms) * 1000;
    while (test_time_us < end) { test_time_us += 1000; loop(); }
}
void send(const char* line) { Serial.feed(std::string(line)+"\n"); advance(5); }
bool coils_held() {
    for (int pin : {2, 3, 4, 5, 8, 9, 10, 11}) if (test_pins[pin]) return true;
    return false;
}
void run_until_stopped_then(unsigned ms, bool expect_held) {
    advance(ms);
    assert(coils_held() == expect_held);
}
// STOP hold lasts 30 s of continuous stop, then coils drop; next DRIVE re-energises.
void hold_scenario() {
    assert(!coils_held());
    send("LIMIT 500");
    for (int i = 0; i < 5; ++i) { send("DRIVE 20 20"); advance(95); }
    const auto stopped_at = left.motor.steps;
    assert(stopped_at > 0 && coils_held());
    send("STOP");
    advance(200);
    assert(controller.left == 0 && coils_held());
    run_until_stopped_then(28000, true);
    Serial.output.clear();
    send("INFO"); // hold=1 stays in INFO: coils are held for the 30 s window.
    assert(Serial.output.find("hold=1") != std::string::npos);
    run_until_stopped_then(1500, true);  // ~29.7 s: still held.
    run_until_stopped_then(1200, false); // >30 s: all eight coil pins low.
    assert(left.motor.mask == 0 && right.motor.mask == 0);
    const auto idle_steps = left.motor.steps;
    advance(500);
    assert(left.motor.steps == idle_steps && !coils_held());
    for (int i = 0; i < 5; ++i) { send("DRIVE 20 20"); advance(95); }
    assert(coils_held() && left.motor.steps > idle_steps);
    send("STOP"); advance(200);
    assert(coils_held()); // New 30 s window starts after the new stop.
    run_until_stopped_then(31000, false);
    Serial.output.clear();
}

// Reviewer fuzz, trimmed: travel <= budget and no stepping once 300 ms have passed
// since the last accepted DRIVE (loop-start time), with the UART time model.
void fuzz(unsigned seed, long iterations) {
    std::mt19937 rng(seed);
    send("STOP"); advance(100);
    double budget = 0; int64_t base_l = 0, base_r = 0; bool have = false;
    const double spm = steps_per_mm;
    for (long it = 0; it < iterations; ++it) {
        test_time_us += (rng() % 20 == 0) ? 1 + rng() % 80000 : 50 + rng() % 1500;
        if (rng() % 100 < 3) {
            std::ostringstream line;
            switch (rng() % 8) {
                case 0: line << "LIMIT " << (0.01 + (rng() % 2000) / 10.0) << "\n"; break;
                case 1: line << "STOP\n"; break;
                case 2: line << "DRIVE " << (int(rng() % 61) - 30) << " " << (int(rng() % 61) - 30) << "\n"; break;
                case 3: line << "DRIVE 30 30\r\n"; break;
                case 4: line << "PING\nINFO\n"; break;
                case 5: line << "INFO\r\nINFO \r\n"; break;
                case 6: line << "STOP\r\n"; break;
                default: line << "DRIVE -30 30\n"; break;
            }
            Serial.feed(line.str());
        }
        const auto before = Serial.output.size();
        const int32_t start_ms = int32_t(millis());
        const auto steps_before = std::make_pair(left.motor.steps, right.motor.steps);
        loop();
        if (Serial.output.find("OK LIMIT", before) != std::string::npos) {
            budget = controller.budget_mm_; base_l = left.motor.steps; base_r = right.motor.steps; have = true;
        }
        if (Serial.output.size() > 100000) Serial.output.clear();
        if (have) {
            const double travel = (std::fabs(double(left.motor.steps - base_l)) +
                                   std::fabs(double(right.motor.steps - base_r))) / (2 * spm);
            assert(travel <= budget + 1e-9);
        }
        if (steps_before != std::make_pair(left.motor.steps, right.motor.steps))
            assert(start_ms - int32_t(controller.last_drive_) < 300);
    }
}

int main() {
    assert(wiring::confirmed && wiring::estop_sense >= 0);
    test_pins[wiring::estop_sense] = HIGH;
    setup();
    assert(Serial.availableForWrite() == 0);
    send("PING"); send("STOP"); send("INFO"); advance(100);
    assert(Serial.output.find("PONG\n") != std::string::npos);
    assert(Serial.output.find("OK STOP\n") != std::string::npos);
    assert(Serial.output.find("STAT ") != std::string::npos);
    assert(Serial.output.find("estop_sense=1") != std::string::npos);
    send("LIMIT 100"); send("DRIVE 20 20"); advance(100);
    assert(left.motor.steps > 0 && controller.left > 0);
    test_pins[wiring::estop_sense] = LOW;
    advance(1);
    assert(controller.left == 0 && controller.right == 0 && controller.estop);
    const auto held = left.motor.steps;
    send("DRIVE 20 20");
    assert(Serial.output.find("ERR estop") != std::string::npos);
    test_pins[wiring::estop_sense] = HIGH;
    advance(200);
    assert(left.motor.steps == held); // Release must not resume.
    Serial.feed(std::string(120, 'x')+"\n"); advance(20);
    assert(Serial.output.find("ERR line too long") != std::string::npos);
    Serial.feed("DRIVE 20 20"); advance(5);
    send("\x18"); send("STOP"); advance(100);
    assert(left.motor.steps == held); // Partial RX discarded; never executed.
    // CRLF line endings must behave like LF for every command, including INFO.
    Serial.output.clear();
    send("INFO\r"); send("INFO \t\r"); send("PING\r"); send("LIMIT 50\r"); send("DRIVE 20 20\r");
    advance(60);
    assert(Serial.output.find("ERR") == std::string::npos);
    assert(Serial.output.find("PONG\n") != std::string::npos && Serial.output.find("OK LIMIT") != std::string::npos);
    assert(Serial.output.find("INFO floorpicker-r4 2 limit=1 hold=1 watchdog_ms=300") != std::string::npos);
    assert(controller.left > 0);
    send("STOP\r"); advance(100);
    assert(Serial.output.find("OK STOP\n") != std::string::npos && controller.left == 0);
    send("INFO x"); assert(Serial.output.find("ERR invalid command") != std::string::npos);
    advance(100); // Reset: INFO x halted.
    // A reply-producing line ends the RX loop: an INFO flood cannot delay the watchdog.
    send("LIMIT 400"); send("DRIVE 25 25");
    const uint64_t drive_us = test_time_us;
    for (int i = 0; i < 300; ++i) Serial.feed("INFO\n");
    uint64_t last_step_us = 0, longest_loop_us = 0;
    for (int i = 0; i < 1000; ++i) {
        test_time_us += 1000;
        const auto steps = left.motor.steps;
        const uint64_t t0 = test_time_us;
        loop();
        longest_loop_us = std::max(longest_loop_us, test_time_us - t0);
        if (left.motor.steps != steps) last_step_us = test_time_us;
        if (Serial.input.size() < 100) Serial.feed("INFO\n");
    }
    assert(last_step_us > drive_us);
    assert(last_step_us - drive_us <= 300000 + longest_loop_us);
    assert(longest_loop_us < 40000);
    Serial.output.clear();
    send("STOP"); advance(31000); // Let the earlier hold expire.
    Serial.output.clear();
    hold_scenario();
    // millis() rollover while holding and releasing.
    test_time_us = (uint64_t(1) << 32) * 1000 - 12000ull * 1000;
    advance(10);
    hold_scenario();
    assert(millis() < 100000); // Time really wrapped during the scenario.
    for (unsigned seed = 1; seed <= 3; ++seed) fuzz(seed, 400000);
    std::cout << "actual main: zero TX capacity, replies, sensed E-stop, release hold, overflow, partial-line cancel, CRLF, INFO flood, 30 s coil hold, fuzz: PASS\n";
}
