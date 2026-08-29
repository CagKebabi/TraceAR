# Changelog

All notable changes to `@tracear/sdk` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-08-29

Steadiness release for multi-target sessions, driven by the first production
integration.

### Added

- **`maxTracked` config option** — cap on simultaneously tracked targets
  (default unlimited). `maxTracked: 1` gives an exclusive session (MindAR's
  `maxTrack: 1` equivalent): once a target is acquired, all other detection
  pauses until it is lost, so exactly one anchor is active at a time and the
  per-frame cost is pure tracking (~2.5 ms). The right mode for "one photo
  plays at a time" products.

### Fixed

- **Trembling during camera motion in multi-target sessions.** While a
  target was tracked, cold-marker scans still ran every 3rd frame; the
  alternation of ~3 ms tracking frames with ~25 ms scan frames made the
  measurement cadence uneven, which read as content trembling while the
  camera moved (still scenes were unaffected). Cold scans now back off to
  every 10th frame while anything is tracked — and don't run at all when
  `maxTracked` slots are full — while the acquire phase (nothing tracked)
  now scans every frame, making first detection faster than 0.2.0. Steady
  10-target cost drops 13 → 6 ms/frame average (native).

## [0.2.0] — 2026-08-29

Multi-target release: a session with many markers (an album of photos, a deck
of cards) now costs roughly the same per frame as a session with one.

### Added

- **Multi-marker pack files.** One `.tracear` file can now hold any number of
  markers. Packs are a pure container: each entry is an unmodified
  single-marker file, so bundling is byte concatenation — adding or removing
  one target never recompiles the others. Anywhere a target is accepted, a
  pack expands in place (marker indices follow the expanded order).
  - CLI: `npx tracear compile a.png b.png c.png -o album.tracear`, and
    `npx tracear pack a.tracear b.tracear -o album.tracear`.
  - Browser: `packMarkers(markers)` in `@tracear/sdk/compiler`.
- `Pipeline::last_detect_indices` diagnostic (which markers attempted
  detection last frame).

### Changed

- **Detection cost is now flat in the number of targets.** Frame features
  (pyramid + FAST + BRIEF) are extracted once per frame and shared by every
  marker's detection, and lost markers are scheduled under a per-frame
  budget: recently-lost targets keep same-frame priority (re-acquire feel is
  unchanged), while long-idle targets take amortized round-robin turns.
  Measured with 10 targets and 1 visible (native, 640×480): 272 → 13 ms per
  frame average. Sessions with 1–2 targets behave exactly as before.
- `targetWidthsMeters` now aligns with expanded marker order (identical to
  before unless you use packs).
- `detectImage` with many targets also shares one feature extraction.

## [0.1.2] — 2026-08-27

### Added

- Error-adaptive catch-up: after a large deviation the pose blend leans toward
  raw measurements and returns to the marker quickly, then settles back into
  filtered smoothness — no more slow, steppy re-anchoring.
- WebCodecs frame path: luma is read straight from the `VideoFrame` Y plane
  where supported, eliminating the canvas readback entirely.

### Fixed

- Safari degrading after long runs: the bitmap fallback path now does its WebGL
  readback into a single reused buffer (zero per-frame allocation).
- Sensor-orientation guard: phones that deliver `VideoFrame`s in landscape
  sensor orientation no longer feed squashed luma to the detector — the worker
  detects the aspect mismatch and falls back to the bitmap path.
- wasm bridge error paths are native-safe, so the core test suite runs
  unmodified outside the browser.

## [0.1.1] — 2026-08-27

### Fixed

- Removed the `development` exports condition from `package.json`. It pointed
  at `src/`, which is not shipped in the tarball, breaking every consumer dev
  server. All consumers now resolve `dist/` unconditionally.

### Documentation

- Vite users: add `optimizeDeps: { exclude: ["@tracear/sdk"] }` so dev-server
  prebundling doesn't break the worker and WASM URLs (documented in the README).

## [0.1.0] — 2026-08-27

Initial release.

- Rust → WASM (+SIMD128) computer-vision core: FAST-9 detection, steered
  BRIEF-256 descriptors, Hamming matching with a scale-aware ratio test,
  normalized DLT + adaptive RANSAC homography.
- Sub-pixel tracking: inverse-compositional Lucas–Kanade over compiled marker
  patches, coarse-to-fine, NCC-validated, with a detect ↔ track state machine.
- 6DoF pose: homography decomposition with LM refinement and online focal
  self-calibration — no camera calibration step for end users.
- Motion-adaptive filtering: One Euro on SE(3) blended with raw measurements by
  instantaneous speed, so a still scene renders frozen and a moving one stays
  glued.
- Worker offload with layered frame paths (WebCodecs → ImageBitmap + WebGL →
  OffscreenCanvas → main-thread canvas), three.js adapter, `.tracear` marker
  format, and an `npx tracear compile` CLI.
- ~95 KB gzipped including the WASM binary.

[Unreleased]: https://github.com/CagKebabi/TraceAR/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.2.1
[0.2.0]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.2.0
[0.1.2]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.1.2
[0.1.1]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.1.1
[0.1.0]: https://github.com/CagKebabi/TraceAR/commits/main
