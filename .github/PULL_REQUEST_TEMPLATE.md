## What

<!-- One or two sentences: what does this PR change, and why? -->

## Related issue

<!-- Fixes #123, or "n/a" -->

## How it was tested

- [ ] `cargo test --release` passes in `core/`
- [ ] `npm run build:sdk` passes (type-check + size budget)
- [ ] Tested on a real phone (device + browser: <!-- e.g. Pixel 8 / Chrome 139 -->)
- [ ] Demo self-test ("Self test (no camera)") still passes

## Checklist

- [ ] Engine conventions respected (coordinate spaces, seeded RNG determinism — see CONTRIBUTING.md)
- [ ] Tracking/pose changes: observed effect on a real device is described above
- [ ] Docs / changelog updated if user-facing
