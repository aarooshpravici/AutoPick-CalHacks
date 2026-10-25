#include "controller.h"
#include <cassert>
#include <string>
#include <iostream>

std::string send(Controller& c, const char* command, uint32_t now = 0) {
    char copy[128]; strcpy(copy, command);
    return c.command(copy, now);
}
int main() {
    Controller c(true, 100, 200, 2000);
    assert(send(c, "PING") == "PONG");
    for (auto command : {"LIFT_HOME", "LIFT_TO 700", "TRAY", "TARE"})
        assert(send(c, command) == "ERR no lift");
    assert(send(c, "DRIVE 100 -100").empty());
    for (uint32_t t = 10; t <= 290; t += 10) c.tick(t, false);
    assert(c.left > 0 && c.right < 0);
    c.tick(300, false); assert(c.left == 0 && c.right == 0);
    send(c, "DRIVE 100 100", 300);
    c.tick(350, false); assert(c.left == 10);
    assert(send(c, "STOP", 350) == "OK STOP");
    c.tick(360, false); assert(c.left == 0 && c.right == 0);
    send(c, "DRIVE 100 100", 400); c.tick(450, false);
    c.tick(451, true); assert(c.left == 0);
    assert(send(c, "DRIVE 100 100", 451) == "ERR estop");
    c.tick(452, false); assert(c.left == 0); // Release never resumes motion.
    for (auto command : {"DRIVE nan 0", "DRIVE inf 0", "DRIVE 101 0", "DRIVE 1x 2",
                         "DRIVE 1 2 extra", "DRIVE 1", "STOP extra", "UNKNOWN"}) {
        assert(send(c, command, 452).find("ERR ") == 0);
        assert(c.left == 0 && c.right == 0);
    }
    Controller disabled(false, 100, 200, 2000);
    assert(send(disabled, "DRIVE 1 1") == "ERR unconfigured motors");
    assert(send(disabled, "STOP") == "OK STOP");
    Controller wrap(true, 100, 200, 2000);
    uint32_t start = UINT32_MAX - 100;
    wrap.tick(start, false); send(wrap, "DRIVE 100 100", start);
    wrap.tick(start + 299, false); assert(wrap.left > 0);
    wrap.tick(start + 300, false); assert(wrap.left == 0);
    std::cout << "protocol, lift absence, ramp, STOP, 300ms watchdog, estop, malformed input, disabled pins, rollover: PASS\n";
}
