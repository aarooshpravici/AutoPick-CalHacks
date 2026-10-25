#pragma once
#include <cstdint>
#include <cstddef>
#include <deque>
#include <string>
constexpr double PI = 3.14159265358979323846;
constexpr int OUTPUT=1, INPUT_PULLUP=2, LOW=0, HIGH=1;
inline uint64_t test_time_us = 0;
inline int test_pins[32] = {};
inline uint32_t millis() { return uint32_t(test_time_us / 1000); }
inline uint32_t micros() { return uint32_t(test_time_us); }
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int value) { test_pins[pin] = value; }
inline int digitalRead(int pin) { return test_pins[pin]; }
struct TestUART {
    std::deque<char> input;
    std::string output;
    void begin(int) {}
    // Renesas UART inherits Print's default. A nonzero stub would hide the bug.
    int availableForWrite() { return 0; }
    int available() { return int(input.size()); }
    int read() { char c=input.front(); input.pop_front(); return c; }
    // UART::write is synchronous: 115200 baud 8N1 is ~87 us per byte of loop time.
    size_t write(const uint8_t* bytes, size_t length) {
        output.append(reinterpret_cast<const char*>(bytes), length);
        test_time_us += 87 * length; return length;
    }
    size_t write(uint8_t byte) { output += char(byte); test_time_us += 87; return 1; }
    void feed(const std::string& text) { for (char c: text) input.push_back(c); }
};
inline TestUART Serial;
