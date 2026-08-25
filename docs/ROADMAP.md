# Roadmap — Phase 1: image tracking engine

Each milestone has acceptance criteria; a milestone is done when its criteria
are met by tests/bench, not by eyeballing. (Device checks are the exception —
they need a human with a phone.)

## M0 — Repo + detection core (native Rust)  ← current

Scaffold, docs, and the full detection pipeline as pure Rust with tests:
image/pyramid, FAST-9, orientation, steered BRIEF, Hamming matcher,
normalized DLT + RANSAC homography, marker compiler (in-memory), detector.

**Accept:** `cargo test` green, including an end-to-end synthetic test:
compiled marker is found in a warped+noisy+brightness-shifted scene with mean
corner error < 3 px; no false positive on a marker-free scene; works at
down-scale ≈ 0.4× and moderate perspective tilt.

## M1 — WASM bridge + camera + 2D overlay demo

`tracear-wasm` bindings crate (wasm-bindgen), `.tracear` binary format +
compiler CLI, TS package skeleton wired to a worker, camera capture,
and a demo page drawing the detected quad over the video (no 3D yet).
First run on real phones (Android Chrome + iOS Safari).

**Accept:** demo detects a printed/on-screen marker live on both platforms;
worker detection < 60 ms/frame on desktop (perf tuning comes later, M4).

## M2 — Frame-to-frame tracking

Tracking patches in the compiled marker, pyramidal inverse-compositional LK,
NCC validation, IRLS homography update, track-quality logic,
detect↔track state machine.

**Accept (bench, synthetic):** static-camera jitter < 0.3 px before filtering;
tracking survives a 500-frame moderate-motion synthetic sequence with < 2
losses; tracking step < 5× faster than detection step.

## M3 — Pose, filtering, three.js

IPPE + LM refinement, focal estimation, One Euro on SE(3) with render-time
prediction, `tracear/three` adapter, 3D demo.

**Accept:** bench jitter (filtered, static) < 0.15 px reprojected; visual
device check: still phone → visually frozen model, fast motion → no swim/lag
complaints at 30 fps.

## M4 — Performance pass + real-device bench

WASM SIMD in hot loops (grayscale, FAST, Hamming, LK), buffer reuse audit,
optional WebGL grayscale path, recorded real-video regression set (user
records; runs in Node), side-by-side MindAR comparison page.

**Accept:** metric table in ARCHITECTURE.md fully met on a mid-range Android
device; MindAR comparison shows lower jitter + lower ms/frame on the same
scenes.

## M5 — SDK polish + npm publish

API freeze, docs site/README with recipes, marker compiler UX
(`npx tracear compile`), error handling (camera permissions, unsupported
browsers), size budget enforcement, CI (tests + bench regression), publish
`tracear@0.x` (free, MIT).

**Accept:** `npm i tracear` + 20-line integration works on a clean project;
bundle < 300 KB gzipped; README quickstart verified on both platforms.

## M6 — Stretch / research

Affine-warped detection for steep tilt, learned features for acquire-only
(WebGPU), multi-target scaling, WebGL/WebGPU compute experiments.

---

# Phase 2 — markerless (after Phase 1)

- **2a:** Android — WebXR `immersive-ar` + hit-test + DOM Overlay integration
  in the same SDK surface (buttons/events over camera = solved there).
- **2b:** iOS — scoped custom tracking (gravity-aligned surface placement via
  DeviceMotion + visual odometry reusing the M2/M3 machinery). Scope is
  explicitly "place on floor/table, acceptable short-session drift" — not
  ARKit-parity SLAM.
