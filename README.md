# Tracear

**High-performance, jitter-free image tracking for the mobile web.**

Tracear is a web AR engine focused on doing one thing extremely well: tracking a
known image target with a phone camera in the browser — smoothly, with low
latency, on both Android and iOS.

> Status: **early development (Phase 1 — image tracking core)**. Not yet published to npm.

## Why another web image tracker?

Existing open-source options (MindAR, AR.js) suffer from visible pose jitter and
performance problems on mid-range phones. Tracear is built from scratch around
three principles:

1. **A lean WASM+SIMD core.** No TensorFlow.js, no generic CV framework overhead.
   Every hot loop is hand-written Rust compiled to WebAssembly with SIMD.
2. **Sub-pixel tracking, not per-frame re-detection.** Detection initializes the
   pose; frame-to-frame tracking refines it with sub-pixel image alignment,
   which is what kills jitter at the source.
3. **Filtering done right.** Pose smoothing on SE(3) (One Euro on translation,
   quaternion-domain filtering on rotation) plus render-time prediction, so the
   result is both stable *and* low-latency.

Quality is measured, not eyeballed: a benchmark harness with synthetic
ground-truth sequences and recorded real-device videos tracks jitter,
accuracy, robustness and speed metrics for every change.

## Repository layout

```
core/                Rust workspace — the tracking engine
  tracear-core/      Pure-Rust CV core (detection, tracking, pose, filtering)
packages/
  tracear/           The npm SDK package (TypeScript, wraps the WASM core)
bench/               Benchmark & regression harness (ground-truth datasets, metrics)
docs/
  ARCHITECTURE.md    Full technical design
  ROADMAP.md         Milestones and acceptance criteria
```

## Development

Rust core (native tests, no browser needed):

```sh
cd core
cargo test
```

Demo (requires [Rust](https://rustup.rs) + [wasm-pack](https://rustwasm.github.io/wasm-pack/) + Node 20):

```sh
npm install
npm run build:wasm   # Rust core -> packages/tracear/wasm
npm run demo         # HTTPS dev server; open it from your phone on the same LAN
```

In the demo: generate (or upload) a marker, then either point the camera at it
or hit **Self test** to run the full detection pipeline on a synthetic frame.

## License

MIT — see [LICENSE](./LICENSE).
