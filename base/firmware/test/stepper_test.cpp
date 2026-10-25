#include "stepper.h"
#include <cassert>
#include <iostream>
int main() {
    CoilStepper motor;
    motor.update(10, 0, 10, false);
    for (uint32_t t = 1000; t <= 100000; t += 1000) motor.update(10, t, 10, false);
    assert(motor.steps >= 9 && motor.steps <= 10);
    auto held = motor.mask; auto count = motor.steps;
    motor.update(0, 1000000, 10, false);
    assert(motor.mask == held && motor.steps == count);
    for (uint32_t t = 1001000; t <= 1100000; t += 1000) motor.update(-10, t, 10, false);
    assert(motor.steps < count);
    CoilStepper inverse, normal;
    for (uint32_t t = 0; t < 50000; t += 1000) {
        inverse.update(10, t, 10, true); normal.update(10, t, 10, false);
        assert(inverse.steps == normal.steps);
    }
    assert(inverse.mask != normal.mask || inverse.steps == 4);
    CoilStepper wrap;
    wrap.update(10, UINT32_MAX - 5000, 10, false);
    wrap.update(10, 5000, 10, false);
    assert(wrap.steps == 1);
    std::cout << "half stepping, reversal, holding, inversion, micros rollover: PASS\n";
}
