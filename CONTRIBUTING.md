# Contributing to TraceAR

Thanks for your interest! Bug reports, tracking-quality reports from real
devices, and pull requests are all welcome.

## Repository layout

| Path | What it is |
|---|---|
| `core/` | Rust workspace: `tracear-core` (pure CV, no browser deps) and `tracear-wasm` (wasm-bindgen bridge) |
| `packages/tracear` | The `@tracear/sdk` npm package (TypeScript) |
| `apps/demo` | Camera demo (Vite) |
| `apps/compare` | TraceAR vs MindAR side-by-side benchmark app |
| `apps/landing` | Static landing page |
| `docs/` | `ARCHITECTURE.md` (design doc) and `ROADMAP.md` (milestones) |

Please read `docs/ARCHITECTURE.md` before making large changes.

## Development setup

Prerequisites: Node 20+, stable Rust with the `wasm32-unknown-unknown` target,
and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/).

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack   # or the installer script

npm install
npm run build:wasm   # builds the WASM bridge into packages/tracear/src/wasm
npm run build:sdk    # dist build (type-check + size budget)
npm run demo         # HTTPS dev server for phone testing over LAN
```

Notes:

- All consumers (dev servers included) resolve the SDK to `dist/` — re-run
  `npm run build:sdk` after every SDK source change.
- For desktop-only work run the demo with `NO_SSL=1` (localhost is a secure
  context on plain HTTP, so no self-signed-cert warning).
- The demo has a **Self test (no camera)** button that runs the full
  worker + WASM detection pipeline on a synthetic scene — use it to verify
  changes without a camera.
- Windows: a non-ASCII path (e.g. a localized desktop folder) can break the
  MinGW linker. Work around it by setting a global ASCII `target-dir` in
  `~/.cargo/config.toml`.

## Tests

```sh
cd core
cargo test              # fast feedback
cargo test --release    # CI runs release mode — always check this before a PR
```

Core tests are pure Rust and need no browser. CI additionally builds the WASM,
the SDK dist, and the demo, and smoke-tests the marker-compiler CLI.

## Engine conventions (do not break)

- Pixel centers sit at integer coordinates; y grows downward.
- Homographies are 3×3 row-major and map **marker level-0 px → frame level-0
  px** unless the variable name says otherwise. Applied as p′ = H·(x, y, 1),
  then divide by w.
- Keypoint coordinates inside per-level code are in that pyramid level's pixel
  space; anything crossing a module boundary is converted to level-0.
- Determinism matters: all randomness (RANSAC sampling, BRIEF pattern,
  synthetic data) goes through the seeded `rng::XorShift64`. Never use system
  RNG or time in the core.
- Scalar Rust first, correctness proven by tests; SIMD comes later and must
  not change results beyond documented tolerances.

## Pull requests

- Keep PRs focused; one logical change per PR.
- Core changes need tests. Tracking/pose changes should state the observed
  effect on a real phone where possible (the compare app gives you a jitter
  number).
- Commit messages follow a light conventional style:
  `feat(core): …`, `fix(sdk): …`, `perf: …`, `docs: …`.

## Reporting tracking quality issues

Jitter, drift, swim, or lost tracking on a specific device are the most
valuable reports we get. Please use the *Tracking quality report* issue
template and include the device, browser, and the numbers from the demo's
stats readout — a short screen recording helps enormously.
