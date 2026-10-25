// Test-only transport adapter for the real firmware Controller. No GPIO or hardware.
#include "controller.h"
#include "stepper.h"
#include <chrono>
#include <iostream>
#include <poll.h>
#include <string>
#include <unistd.h>

int main() {
    Controller controller(true, 30, 30, 300);
    const auto start = std::chrono::steady_clock::now();
    uint32_t last_status = 0;
    CoilStepper left, right;
    const float steps_per_mm = 4096 / (3.14159265f * 71);
    unsigned pending_stops = 0;
    std::string line;
    while (true) {
        const auto now = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count());
        controller.wheel_steps(left.steps, right.steps, steps_per_mm);
        controller.tick(now, false);
        pollfd input{STDIN_FILENO, POLLIN, 0};
        if (poll(&input, 1, 1) > 0 && (input.revents & POLLIN)) {
            char buffer[32];
            const auto count = read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count <= 0) return 0;
            for (int i = 0; i < count; ++i) {
                if (buffer[i] == '\n') {
                    std::cerr << line << std::endl;
                    const char* response = controller.command(line.data(), now);
                    if (!strcmp(response, "OK STOP")) ++pending_stops;
                    else if (*response) std::cout << response << std::endl;
                    line.clear();
                } else if (line.size() < 95) line += buffer[i];
                else return 2;
            }
        }
        left.update(controller.left, now * 1000, steps_per_mm, false);
        right.update(controller.right, now * 1000, steps_per_mm, false);
        if (pending_stops && controller.left == 0 && controller.right == 0) {
            std::cout << "OK STOP" << std::endl;
            --pending_stops;
        }
        if (now - last_status >= 100) {
            std::cout << "STAT " << left.steps << " " << right.steps << " 0 0 0" << std::endl;
            if (controller.limit_reached) std::cout << "LIMIT REACHED" << std::endl;
            last_status = now;
        }
    }
}
