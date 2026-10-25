# Floor Picker UI review

Captured from localhost in simulation mode. Photos here are browser screenshots,
not camera feeds. The handoff screenshot uses a simulated status response.

## Before

![Original desktop page](before-desktop.png)

## After

![Polished desktop page](after-desktop.png)

## Phone layout

![Polished phone page](after-mobile.png)

## Handoff state

![Handoff state preview](handoff-desktop.png)

## Validation

- Existing server/request suite: 164 passed, 1 skipped.
- Native macOS WebKit: simulation indicator, a real request to the simulated
  backend, STOP, a mocked disconnected response, handoff and camera empty states.
- 320 px and 390 px layouts: no horizontal overflow; STOP stays visible and at
  least 48 px tall after scrolling. Desktop captured at 1440 × 1080 CSS pixels.
- No hardware was connected or moved during these checks.

![Narrow phone layout](after-small-mobile.png)
