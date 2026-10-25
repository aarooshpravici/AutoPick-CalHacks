#include "Arduino.h"
#include "../base/firmware/src/main.cpp"
#include <cassert>
#include <iostream>

void advance(unsigned ms) {
    for (unsigned i=0; i<ms; ++i) { test_time_us += 1000; loop(); }
}
void send(const char* line) { Serial.feed(std::string(line)+"\n"); advance(5); }
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
    std::cout << "actual main: zero TX capacity, replies, sensed E-stop, release hold, overflow, partial-line cancel: PASS\n";
}
