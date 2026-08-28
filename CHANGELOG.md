# Changelog

All notable changes to `@tracear/sdk` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/CagKebabi/TraceAR/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.1.2
[0.1.1]: https://github.com/CagKebabi/TraceAR/releases/tag/v0.1.1
[0.1.0]: https://github.com/CagKebabi/TraceAR/commits/main
