#include <Arduino.h>
#include "controller.h"
#include "pinmap.h"
#include "stepper.h"

Controller controller(wiring::confirmed, wiring::max_speed_mm_s,
                      wiring::acceleration_mm_s2, wiring::stop_acceleration_mm_s2);
const float steps_per_mm = wiring::half_steps_per_rev / (PI * wiring::wheel_diameter_mm);

struct Wheel {
    const int* pins;
    bool inverted;
    CoilStepper motor;
    void begin() {
        for (int i = 0; i < 4; ++i) { pinMode(pins[i], OUTPUT); digitalWrite(pins[i], LOW); }
    }
    void update(float speed, uint32_t now, bool emergency) {
        // Holding means retaining the last phase, never releasing coils on STOP.
        if (motor.update(emergency ? 0 : speed, now, steps_per_mm, inverted))
            for (int i = 0; i < 4; ++i) digitalWrite(pins[i], (motor.mask >> i) & 1);
    }
};
Wheel left{wiring::left_coils, wiring::invert_left, {}};
Wheel right{wiring::right_coils, wiring::invert_right, {}};
char input[96];
size_t used = 0;
bool overflow = false;
uint32_t last_status = 0;
unsigned pending_stops = 0;

void reply(const char* message) {
    // Uno R4 NO_USB Serial is UART: inherited availableForWrite() always returns
    // zero. UART::write transmits synchronously (~87 us/byte at 115200 baud).
    // RX is capped at 32 bytes per loop; replies/status are bounded short lines.
    const size_t length = strlen(message);
    if (length) {
        Serial.write(reinterpret_cast<const uint8_t*>(message), length);
        Serial.write('\n');
    }
}

void setup() {
    Serial.begin(115200);
    if (wiring::confirmed) {
        left.begin(); right.begin();
        if (wiring::estop_sense >= 0) pinMode(wiring::estop_sense, INPUT_PULLUP);
    }
}

void loop() {
    const uint32_t now = millis();
    const bool emergency = wiring::confirmed && wiring::estop_sense >= 0 &&
        (digitalRead(wiring::estop_sense) == (wiring::estop_active_low ? LOW : HIGH));
    controller.wheel_steps(left.motor.steps, right.motor.steps, steps_per_mm);
    controller.tick(now, emergency);
    // Bound RX work per loop so an input flood cannot starve the watchdog.
    for (int n = 0; n < 32 && Serial.available(); ++n) {
        const char c = char(Serial.read());
        if (c == '\x18') { // CAN: discard a partial monitor line without executing it.
            controller.halt(); used = 0; overflow = false;
        } else if (c == '\n') {
            if (overflow) { controller.halt(); reply("ERR line too long"); }
            else {
                input[used] = '\0';
                if (!strcmp(input, "INFO")) {
                    char info[192];
                    const long speed_um_s = lroundf(wiring::max_speed_mm_s * 1000);
                    snprintf(info, sizeof(info), "INFO floorpicker-r4 2 limit=1 hold=1 watchdog_ms=300 wiring=%d estop_sense=%d max_mm_s=%ld.%03ld config=%s",
                             int(wiring::confirmed), int(wiring::estop_sense >= 0),
                             speed_um_s / 1000, speed_um_s % 1000, wiring::config_id);
                    reply(info);
                    used = 0; overflow = false;
                    continue;
                }
                const char* response = controller.command(input, now);
                if (!strcmp(response, "OK STOP")) {
                    if (pending_stops < 64) ++pending_stops;
                    else reply("ERR too many STOP requests");
                } else reply(response);
            }
            used = 0; overflow = false;
        } else if (c == '\0' || (static_cast<unsigned char>(c) < 32 && c != '\r' && c != '\t')) {
            overflow = true; controller.halt();
        } else if (used + 1 < sizeof(input) && !overflow) input[used++] = c;
        else { overflow = true; controller.halt(); }
    }
    if (wiring::confirmed) {
        left.update(controller.left, micros(), emergency);
        right.update(controller.right, micros(), emergency);
    }
    if (pending_stops && controller.left == 0 && controller.right == 0) {
        reply("OK STOP");
        --pending_stops;
    }
    if (uint32_t(now - last_status) >= 100) {
        last_status = now;
        char status[96];
        snprintf(status, sizeof(status), "STAT %ld %ld 0 0 %d", long(left.motor.steps), long(right.motor.steps), int(emergency));
        reply(status);
        if (controller.limit_reached) reply("LIMIT REACHED");
    }
}
