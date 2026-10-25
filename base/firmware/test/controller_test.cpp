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
    assert(send(c, "DRIVE 1 1") == "ERR travel limit required");
    assert(send(c, "LIMIT 1000") == "OK LIMIT");
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
    assert(send(wrap, "LIMIT 1000") == "OK LIMIT");
    uint32_t start = UINT32_MAX - 100;
    wrap.tick(start, false); send(wrap, "DRIVE 100 100", start);
    wrap.tick(start + 299, false); assert(wrap.left > 0);
    wrap.tick(start + 300, false); assert(wrap.left == 0);
    Controller bounded(true, 100, 200, 2000);
    bounded.wheel_steps(100, -100, 10);
    for (auto bad : {"LIMIT 0", "LIMIT -1", "LIMIT nan", "LIMIT inf", "LIMIT 6097", "LIMIT 2 extra"})
        assert(send(bounded, bad).find("ERR") == 0);
    assert(send(bounded, "LIMIT 1") == "OK LIMIT");
    send(bounded, "DRIVE 10 -10"); bounded.tick(50, false);
    assert(send(bounded, "LIMIT 10") == "ERR invalid travel limit");
    // Turning counts and STOP cannot reset the remaining budget.
    bounded.wheel_steps(105, -105, 10);
    send(bounded, "STOP"); bounded.tick(100, false);
    bounded.wheel_steps(109, -109, 10);
    assert(bounded.limit_reached && bounded.left == 0 && bounded.right == 0);
    assert(send(bounded, "DRIVE 1 1") == "LIMIT REACHED");
    assert(send(bounded, "LIMIT 2") == "OK LIMIT");
    assert(!bounded.limit_reached);
    assert(send(bounded, "DRIVE -1 -1").empty());
    // Reversal consumes travel; signed cancellation cannot extend the budget.
    bounded.wheel_steps(99, -119, 10);
    bounded.wheel_steps(109, -109, 10);
    assert(bounded.limit_reached);
    // Plain decimal only: no hex floats, inf/nan words, or stray characters.
    Controller numbers(true, 100, 200, 2000);
    for (auto bad : {"LIMIT 0x1p3", "LIMIT 0X10", "LIMIT 1p3", "LIMIT infinity", "LIMIT INF", "LIMIT NaN",
                     "LIMIT 1e", "LIMIT e5", "LIMIT .", "LIMIT +", "LIMIT 1e+", "LIMIT 1..2", "LIMIT 1e1.5",
                     "LIMIT --1", "LIMIT 1f", "LIMIT 1,5", "LIMIT 1e999"})
        assert(send(numbers, bad) == "ERR invalid travel limit");
    for (auto good : {"LIMIT 5", "LIMIT +5", "LIMIT 5.", "LIMIT .5", "LIMIT 0.5e1", "LIMIT 5E0", "LIMIT 50e-1"})
        assert(send(numbers, good) == "OK LIMIT");
    for (auto bad : {"DRIVE 0x10 0", "DRIVE 0 0x1p1", "DRIVE nan 0", "DRIVE 1e 0"})
        assert(send(numbers, bad) == "ERR invalid speed");
    assert(send(numbers, "DRIVE 1.5e1 -.5").empty());
    // CRLF endings are tolerated (controller strips them; main strips before INFO).
    assert(send(numbers, "PING\r") == "PONG" && send(numbers, "STOP\r") == "OK STOP");
    assert(send(numbers, "LIMIT 7\r") == "OK LIMIT" && send(numbers, "DRIVE 1 1\r").empty());
    std::cout << "protocol, lift absence, ramp, STOP, 300ms watchdog, estop, malformed input, disabled pins, rollover: PASS\n";
}
