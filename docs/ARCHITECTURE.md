# Tracear architecture

Tracear is a web-native image tracking engine. This document is the technical
design for Phase 1 (image tracking). It explains what each stage does, which
algorithm was chosen and why, and how the pieces map onto the web platform.

## Goals and non-goals

**Goals**

- Track one or more known planar image targets from a phone camera in the
  browser at 30 fps on mid-range devices (Android Chrome, iOS Safari 16.4+).
- Visibly less jitter than MindAR: sub-pixel-stable pose while the phone is
  held still, no swimming while it moves.
- Ship as a small, dependency-free npm package (`tracear`) with a
  framework-agnostic core plus an optional three.js adapter.
- Every quality claim backed by the benchmark harness (see `bench/`).

**Non-goals (Phase 1)**

- Markerless world tracking (Phase 2/3 — see ROADMAP).
- Non-planar targets, curved surfaces, faces.
- Native (non-web) runtimes. The core is portable Rust, so native reuse stays
  possible, but web is the only supported target for now.

## System overview

```
          ┌────────────────────────────  main thread  ─────────────────────────┐
          │  getUserMedia → <video> → requestVideoFrameCallback                │
          │        │                                                           │
          │        │ VideoFrame (GPU)                                          │
          │        ▼                                                           │
          │  frame grabber (WebGL downscale+grayscale OR VideoFrame.copyTo)    │
          │        │ gray Uint8Array (transferable / SharedArrayBuffer)        │
          └────────┼───────────────────────────────────────────────────────────┘
                   ▼
          ┌──── worker ────────────────────────────────────────────────────────┐
          │   WASM core (Rust, SIMD)                                           │
          │   ┌──────────────┐   not tracking   ┌─────────────────────────┐    │
          │   │  DETECTION   │ ───────────────▶ │  initial H, pose        │    │
          │   │ FAST+BRIEF+  │                  └───────────┬─────────────┘    │
          │   │ match+RANSAC │ ◀── track lost ──────────────│                  │
          │   └──────────────┘                  ┌───────────▼─────────────┐    │
          │                                     │  TRACKING (per frame)   │    │
          │                                     │  pyramidal LK patches   │    │
          │                                     │  + robust H refinement  │    │
          │                                     │  + NCC validation       │    │
          │                                     └───────────┬─────────────┘    │
          │                                     ┌───────────▼─────────────┐    │
          │                                     │  POSE (IPPE + LM)       │    │
          │                                     └───────────┬─────────────┘    │
          └─────────────────────────────────────────────────┼──────────────────┘
                   raw pose + timestamp + quality           ▼
          ┌────────────────────────────  main thread  ─────────────────────────┐
          │  FILTER + PREDICT (One Euro on SE(3), extrapolate to rAF time)     │
          │        ▼                                                           │
          │  SDK events / three.js anchor update                               │
          └────────────────────────────────────────────────────────────────────┘
```

Two key architectural decisions:

1. **Detection and tracking are separate paths.** Detection (expensive, robust)
   runs only to acquire or re-acquire the target. Tracking (cheap, sub-pixel)
   runs every frame. MindAR-style pipelines that lean on detection every frame
   pay both a performance and a jitter cost; sub-pixel refinement is where
   smoothness comes from.
2. **Filtering runs on the main thread at render cadence,** decoupled from the
   (possibly slower / jittery-latency) worker results. The renderer never waits
   for the tracker: it filters the latest measurement and predicts to the
   current frame's presentation time.

## Pipeline stages

### 1. Marker compiler (offline / build-time)

Input: target image. Output: a compact binary (`.tracear`) with everything the
runtime needs, so no feature extraction happens on the marker at runtime.

- Build a scale series of the marker: factor `1/1.26` (≈ 2^(1/3)) per step,
  from native size down to min side ≈ 64 px. The runtime camera pyramid uses
  factor 2; the marker's 1.26-step series guarantees any observed scale is
  within ~13% of a compiled scale, which BRIEF matching tolerates.
- Per scale: FAST-9 corners → uniform grid selection (top-K per cell, by
  corner score) → orientation (intensity centroid) → 256-bit steered BRIEF
  descriptors on a blurred copy. Feature positions are stored in marker
  level-0 pixel coordinates.
- Later (M2): also store tracking patches — small high-gradient templates used
  by the frame-to-frame tracker.

The compiler is the same Rust code as the runtime (one crate), compiled to a
CLI tool and a browser/WASM build, so compiled markers are bit-identical from
either path.

### 2. Detection

Runs on the camera frame until the target is found (and in the background at a
low duty cycle while tracking, for multi-target and recovery).

- **Frame pyramid:** grayscale camera frame (long side capped ≈ 640 px),
  half-scale levels down to min side ≈ 80 px.
- **FAST-9** corners per level with a 16-bit ring trick for the contiguous-arc
  test, score = clipped sum of absolute differences, 3×3 non-max suppression.
- **Uniform selection:** grid bucketing, top-K per cell — feature spread
  matters more than raw count for homography conditioning.
- **Orientation + steered BRIEF (256 bit)** on a box-blurred copy of each
  level (BRIEF without pre-smoothing is noise-dominated).
- **Matching:** brute-force Hamming (XOR + popcount — SIMD-friendly), with a
  scale-aware ratio test: the second-best candidate must be spatially distinct
  in marker space, otherwise the same physical feature at an adjacent compiled
  scale would veto its own match.
- **Homography:** RANSAC (adaptive iteration count, degenerate-sample
  rejection) over matches, minimal 4-point normalized DLT, final least-squares
  refit on inliers. Acceptance requires a minimum inlier count and ratio.

Why classical features and not a learned detector (SuperPoint et al.)? A
learned detector costs tens of ms per frame on mobile web even with WebGPU,
and detection is not the jitter bottleneck — tracking is. Classical detection
is robust enough for planar textured targets and leaves the frame budget to
the tracker. Revisit in M6 (detection-only, on-acquire).

### 3. Tracking (the jitter killer) — M2

Once H is known, per frame:

- Predict H for the new frame (constant-velocity on the filter state).
- Select N (~25–40) precompiled marker patches currently visible and well
  distributed in the frame.
- For each patch: warp the template by the predicted H, run pyramidal
  inverse-compositional Lucas-Kanade to sub-pixel convergence, validate with
  NCC (normalized cross-correlation) and drop weak/occluded patches.
- Update H from surviving correspondences with IRLS (Huber weights), not plain
  least squares — a few bad patches must not wobble the whole pose.
- Track quality = surviving-patch ratio + NCC stats. Below threshold →
  re-detection.

Sub-pixel LK on warped templates is the single most important difference from
re-detection-based pipelines: corner detectors quantize, LK converges to
~0.05–0.1 px, and pose noise scales directly with point noise.

### 4. Pose estimation — M3

- Camera intrinsics: estimated focal from typical mobile FOV (~60–70°) as a
  starting point, refined online from the homography sequence (planar targets
  give a usable focal estimate over time). No user calibration step.
- Homography → pose with **IPPE** (fast, analytic, gives both solutions and
  their errors — the ambiguity matters at near-frontal views), then
  Levenberg-Marquardt refinement minimizing reprojection error.
- Output: `[R|t]` with marker physical size normalized (marker width = 1 unit
  by default; SDK lets you set physical width in meters).

### 5. Filtering & prediction — M3

- **One Euro filter** on translation (velocity-adaptive cutoff: still hand →
  strong smoothing, fast motion → low latency).
- Rotation filtered in quaternion space (slerp-based One Euro / log-space
  filtering on SO(3)) — never filter Euler angles or matrix entries.
- Measurements are timestamped with the camera frame's capture time;
  the render loop queries `pose(t_render)` which filters + extrapolates.
  This absorbs worker scheduling jitter and camera→display latency.

### 6. Web integration (SDK) — M1/M5

- **Frame acquisition:** `requestVideoFrameCallback` when available (Safari
  15.4+, Chrome), fallback to rAF sampling. Grayscale conversion + downscale
  via WebGL fragment shader with `readPixels` into a reusable buffer (fast
  path), or `VideoFrame.copyTo` + WASM conversion (fallback).
- **Threading:** the CV pipeline runs in a Web Worker. With COOP/COEP
  (crossOriginIsolated) pages, frames go through a SharedArrayBuffer ring;
  otherwise transferable ArrayBuffers. WASM threads are *not* required — the
  design targets a single worker so the SDK works on any static host.
- **Public API sketch:**

```ts
import { Tracear } from 'tracear';

const tracker = await Tracear.create({
  container: document.querySelector('#ar'),   // manages <video> + overlay sizing
  targets: ['/markers/poster.tracear'],
});
tracker.on('targetFound', ({ index }) => { /* ... */ });
tracker.on('targetLost',  ({ index }) => { /* ... */ });
tracker.on('update', ({ index, pose, homography, quality }) => { /* raw access */ });
await tracker.start();
```

- `tracear/three`: optional adapter exposing a `THREE.Group` anchor per target
  (auto camera FOV setup, visibility toggling). Core stays renderer-agnostic.
- `tracear/compiler`: marker compilation in the browser or via
  `npx tracear compile poster.png` (Node + WASM).

## Determinism & testing strategy

- All randomness flows through a seeded xorshift RNG; a given input always
  produces the same output. This makes regressions bisectable and benches
  reproducible.
- The core is pure Rust with no I/O, tested natively (`cargo test`) — the
  browser is only needed for integration, not algorithm work.
- Synthetic ground truth: textured images warped under known homographies with
  noise/blur/brightness perturbations. Detection accuracy and (later) tracking
  jitter are asserted against known poses in unit tests; the full-trajectory
  version lives in `bench/`.

## Metrics (bench harness)

| Metric | Definition | Phase-1 target |
|---|---|---|
| Jitter (static) | std-dev of reprojected marker-corner px over a static-camera segment | < 0.15 px |
| Accuracy | mean corner reprojection error vs ground truth | < 1.5 px |
| Detection rate | fraction of ground-truth-visible frames with successful acquire | > 95% |
| Tracking loss | losses per 1000 frames on moderate-motion sequences | < 5 |
| CPU time | worker ms/frame at 640 px input, mid-range phone | < 8 ms tracking, < 30 ms detection |
| SDK size | gzipped JS + WASM | < 300 KB |

## Risks / honest unknowns

- **iOS camera quirks** (resolution caps, `requestVideoFrameCallback` timing,
  auto-exposure swings) — mitigated by the M1 device-smoke-test demo, early.
- **Focal estimation quality** affects pose realism (not stability). Fallback:
  per-device FOV table + manual override in the SDK config.
- **BRIEF under strong tilt** (> ~55°) will degrade detection; acceptable for
  v1, affine-warped detection retry is a known extension.
- Real-device jitter has sources outside our control (rolling shutter, AE
  flicker); the filter layer is designed to absorb what the tracker can't.
