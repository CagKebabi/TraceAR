# TraceAR

[![npm](https://img.shields.io/npm/v/%40tracear%2Fsdk?label=%40tracear%2Fsdk&color=2a6df4)](https://www.npmjs.com/package/@tracear/sdk)
[![CI](https://github.com/CagKebabi/TraceAR/actions/workflows/ci.yml/badge.svg)](https://github.com/CagKebabi/TraceAR/actions/workflows/ci.yml)
[![size](https://img.shields.io/badge/SDK-95%20KB%20gzipped-39d98a)](packages/tracear)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**High-performance, jitter-free image tracking for the mobile web.**

TraceAR is a web AR engine focused on doing one thing extremely well: tracking a
known image target with a phone camera in the browser — smoothly, with low
latency, on both Android and iOS.

## Try it now

No install needed — scan with your phone (or open on any device with a camera):

| [**Live demo →**](https://cagkebabi.github.io/TraceAR/demo/) | [**Side-by-side with MindAR →**](https://cagkebabi.github.io/TraceAR/compare/) |
|---|---|
| <img src="apps/landing/assets/qr-demo.png" width="150" alt="QR: live demo" /> | <img src="apps/landing/assets/qr-compare.png" width="150" alt="QR: comparison" /> |

```sh
npm i @tracear/sdk
```

```ts
import { Tracear } from "@tracear/sdk";

const tracker = await Tracear.create({ container, targets: ["/poster.tracear"] });
tracker.on("update", (e) => {/* filtered 6DoF pose + homography */});
await tracker.start();
```

Full quickstart, three.js recipe and the marker compiler:
**[packages/tracear →](packages/tracear#readme)**

## How it compares

[MindAR](https://github.com/hiukim/mind-ar-js) is the pioneering open-source
web image tracker and a big part of why web AR exists at all — TraceAR simply
makes different engineering trade-offs (a hand-written WASM core instead of a
general ML runtime, sub-pixel tracking instead of per-frame detection). The
numbers below were measured on the same phone, same marker, with an identical
metric — and you can reproduce them on your own device with the side-by-side
app in [`apps/compare`](apps/compare):

| | **TraceAR** | MindAR |
|---|---|---|
| median jitter (640 px frame) | 1.90 px | 5.47 px |
| p90 jitter | 5.8 px | 8.3 px |
| CV time / frame (tracking) | ~2.5 ms | n/a (not exposed) |
| marker compile (512 px) | 0.1 s / 169 KB | 1.4 s / 418 KB |
| SDK size (gzipped, incl. WASM) | ~95 KB | ~370 KB (tfjs runtime) |

> Phase 1 (image tracking) complete; APIs may still move before 1.0.

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
