# Performance & browsers

## Measured numbers

From a real mid-range Android phone, measured with the
[comparison app](https://cagkebabi.github.io/TraceAR/compare/)'s
motion-immune jitter metric (second differences of the marker center,
normalized to a 640 px frame — deliberate camera motion cancels out, only
shake counts):

| Metric | TraceAR | MindAR (same device) |
|---|---|---|
| Median jitter | **1.90 px** | 5.47 px |
| p90 jitter | **5.8 px** | 8.3 px |
| CV time per tracked frame | **~2.5 ms** | n/a (not exposed) |
| Marker compile (512 px) | **0.1 s · ~170 KB** | 1.4 s · ~420 KB |
| SDK size (gzipped) | **~95 KB** | ~370 KB (tfjs runtime) |

[MindAR](https://github.com/hiukim/mind-ar-js) is the pioneering open-source
web image tracker and takes a different engineering approach; the comparison
app exists so the trade-offs can be measured fairly on your own device, with
the identical metric for both engines.

## How the budget is spent

- All computer vision runs in a **worker**, in Rust compiled to
  **WASM + SIMD**. The main thread only pumps frames and renders.
- Frames reach the worker through the fastest path the browser supports,
  falling back automatically:
  1. **WebCodecs `VideoFrame`** — the worker reads the camera's Y plane
     directly; no canvas anywhere.
  2. **`createImageBitmap` + worker-side WebGL readback** — the GPU resize
     stays on the GPU; readback happens off the main thread into a reused
     buffer.
  3. **OffscreenCanvas / main-thread canvas** — the compatible slow path.
- Busy frames are **dropped, never queued**: results stay real-time and
  latency can't accumulate.

## Tuning

- **`maxProcessSize`** (default 640): the processing resolution. 640 is the
  sweet spot in our testing; lowering it buys speed on very weak devices at
  the cost of tracking range.
- **Marker choice** matters more than any setting — see
  [Markers](/guide/markers).
- Keep your render loop light: the tracker leaves most of the main thread
  free, so if the scene stutters, profile your three.js scene first.

## Many targets {#many-targets}

Since 0.2.0, per-frame cost is flat in the number of loaded targets. Frame
features are extracted once and shared by every marker's detection, and lost
targets are scheduled under a per-frame budget: a *recently lost* target
keeps same-frame priority (so re-acquiring the photo the user is pointing at
feels instant), while long-idle targets take amortized round-robin turns —
their only cost is a slightly longer time-to-first-acquire (a few hundred
milliseconds with 10 targets).

Measured natively with 10 targets loaded and 1 visible (640×480): ~272 ms per
frame with naive per-marker detection versus **~13 ms average** with the
scheduler. Each additional *visible* target still adds its own (cheap)
tracking cost.

## Browser support

| Platform | Status |
|---|---|
| Android Chrome | ✅ primary target, device-tested |
| iOS Safari 16.4+ | ✅ primary target, device-tested |
| Desktop Chrome / Edge | ✅ works (webcam) |
| Desktop Safari 16.4+ / Firefox | ✅ expected to work via the fallback paths |

Requirements: a secure context (HTTPS or localhost), workers, and
WebAssembly SIMD — which sets the Safari 16.4 floor. WebCodecs and
OffscreenCanvas are used opportunistically, never required.

::: tip Report your device
Real-device reports are the most valuable feedback this project gets. If
tracking is worse on your phone than the numbers above, please open a
[tracking quality report](https://github.com/CagKebabi/TraceAR/issues/new/choose).
:::
