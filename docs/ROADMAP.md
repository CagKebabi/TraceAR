# Roadmap — Phase 1: image tracking engine

Each milestone has acceptance criteria; a milestone is done when its criteria
are met by tests/bench, not by eyeballing. (Device checks are the exception —
they need a human with a phone.)

## M0 — Repo + detection core (native Rust)  ✅ done (2026-08-26)

Scaffold, docs, and the full detection pipeline as pure Rust with tests:
image/pyramid, FAST-9, orientation, steered BRIEF, Hamming matcher,
normalized DLT + RANSAC homography, marker compiler (in-memory), detector.

**Accept:** `cargo test` green, including an end-to-end synthetic test:
compiled marker is found in a warped+noisy+brightness-shifted scene with mean
corner error < 3 px; no false positive on a marker-free scene; works at
down-scale ≈ 0.4× and moderate perspective tilt.

## M1 — WASM bridge + camera + 2D overlay demo  ✅ done (2026-08-26)

`tracear-wasm` bindings crate (wasm-bindgen), `.tracear` binary format +
in-browser compiler (`tracear/compiler`; the Node CLI moved to M5), TS
package wired to a worker, camera capture, `detectImage()` one-shot API,
and a demo page drawing the detected quad over the video (no 3D yet).
First run on real phones (Android Chrome + iOS Safari).

**Accept:** demo detects a printed/on-screen marker live on both platforms;
worker detection < 60 ms/frame on desktop (perf tuning comes later, M4).
Status: browser self-test 47 ms / 162 inliers / 0.5 px on desktop Chrome;
phone test passed — stable quad lock at 37-39 ms/frame on device.

## M2 — Frame-to-frame tracking  ✅ done (2026-08-26)

Tracking patches in the compiled marker (21x21 templates at factor-2 levels,
.tracear v2), translation-only inverse-compositional LK on re-warped
templates, NCC validation, Huber-IRLS homography update, track-quality
logic, detect↔track state machine (`pipeline.rs`).

**Accept (bench, synthetic):** static-camera jitter < 0.3 px before filtering;
tracking survives a 500-frame moderate-motion synthetic sequence with < 2
losses; tracking step ≥ 5× faster than detection step.
Measured (`cargo run --release --example bench_track`): static jitter
**0.019 px**; motion **0 losses**, 499/500 tracked, 0.05 px mean corner
error; track 3.45 ms vs detect 33.9 ms = **9.8×**. All targets exceeded.

## M3 — Pose, filtering, three.js  ✅ done (2026-08-26)

Zhang decomposition + LM refinement (previous pose seeds refinement to stay
on the same planar-ambiguity branch), online focal self-calibration
(median-filtered Zhang constraints, typical-FOV default), One Euro on SE(3)
(zero-mean-derivative variant, quaternion-domain rotation filtering) with
velocity outputs consumed by `poseAt()` render-time prediction,
`tracear/three` adapter (intrinsics-driven projection, per-target anchors),
3D demo (cube + axes).

**Accept:** bench jitter (filtered, static) < 0.15 px reprojected; visual
device check: still phone → visually frozen model, fast motion → no swim/lag
complaints at 30 fps.
Status: session test asserts filtered reprojected jitter < 0.15 px (raw
tracker jitter is already 0.017 px); pose recovery exact to 0.1 deg on
synthetic homographies; focal estimator converges within 5% (and measured
the real device correctly through a portrait stream). Device iterations
led to the speed-adaptive raw/filtered rendering blend + orientation-
invariant focal ratio; device check passed ("frozen when still" achieved,
motion glue acceptable — further polish rides on M4 real-device tuning).

## M4 — Performance pass + real-device bench  🟡 in progress

WASM SIMD in hot loops (grayscale, FAST, Hamming, LK), buffer reuse audit,
optional WebGL grayscale path, recorded real-video regression set (user
records; runs in Node), side-by-side MindAR comparison page.

**Accept:** metric table in ARCHITECTURE.md fully met on a mid-range Android
device; MindAR comparison shows lower jitter + lower ms/frame on the same
scenes.
Progress (2026-08-26): profile-driven scalar pass — running-sum box blur
(bit-identical, guarded by a naive-reference test), FAST candidate-list NMS
+ bounds-check elision, row-sliced orientation, BRIEF unchecked sampling —
native detection 34 -> 25 ms; wasm32 builds with +simd128
(core/.cargo/config.toml). Desktop-browser detection 41-58 -> ~21 ms warm
(~2x). Remaining: MindAR comparison page, real-video set, device metric
table.

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
