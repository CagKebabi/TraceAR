# Tracear bench harness

Measurement-driven development: every quality claim (jitter, accuracy,
robustness, speed) is a number produced here, compared against the targets in
`docs/ARCHITECTURE.md` and against previous runs.

## Planned structure (lands with M2/M4)

- `datasets/synthetic/` — generated ground-truth sequences: a compiled marker
  rendered along scripted camera trajectories (static hold, slow pan, fast
  shake, tilt sweep, scale sweep) with noise/blur/exposure perturbations.
  Fully deterministic (seeded) — regenerated, not committed.
- `datasets/real/` — short phone-recorded clips (Android + iOS) of printed and
  on-screen markers. No ground truth; used for tracking-loss and jitter-proxy
  metrics and for regression comparison.
- `src/` — Node runner: feeds sequences through the same WASM core the browser
  uses, emits a metrics table (JSON + markdown) into `out/`.

## Metrics

| Metric | Definition |
|---|---|
| jitter_static_px | std-dev of reprojected marker-corner position over static segments |
| accuracy_px | mean corner reprojection error vs ground truth |
| detection_rate | acquired frames / ground-truth-visible frames |
| losses_per_1k | tracking losses per 1000 frames, moderate motion |
| ms_detect / ms_track | per-stage worker time at 640 px input |

Native quick-check exists already in the core:
`cd core && cargo run --release --example bench_detect`.
