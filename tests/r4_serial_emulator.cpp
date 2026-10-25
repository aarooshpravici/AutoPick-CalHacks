// POSIX transport around the actual production setup()/loop(), with fake GPIO/UART.
#include "Arduino.h"
#include "../base/firmware/src/main.cpp"
#include <chrono>
#include <iostream>
#include <poll.h>
#include <unistd.h>

int main() {
    setup();
    const auto start = std::chrono::steady_clock::now();
    std::string command;
    while (true) {
        test_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        pollfd input{STDIN_FILENO, POLLIN, 0};
        const int ready = poll(&input, 1, 1);
        if (input.revents & (POLLHUP | POLLERR | POLLNVAL)) return 0;
        if (ready > 0 && (input.revents & POLLIN)) {
            char buffer[32];
            const auto count = read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count <= 0) return 0;
            for (int i=0; i<count; ++i) {
                Serial.input.push_back(buffer[i]);
                if (buffer[i]=='\n') { std::cerr << command << std::endl; command.clear(); }
                else command += buffer[i];
            }
        }
        loop();
        if (!Serial.output.empty()) {
            std::cout << Serial.output << std::flush;
            Serial.output.clear();
        }
    }
}
