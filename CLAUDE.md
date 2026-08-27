# Tracear — project notes for Claude

Web AR image-tracking engine, to be published as a **free npm SDK** (`tracear`).
Goal: beat MindAR on jitter + performance on mobile web (Android Chrome, iOS Safari).

- Communicate with the user in **Turkish**. Code, comments, and docs are in **English** (public OSS project).
- Design doc: `docs/ARCHITECTURE.md`. Milestones: `docs/ROADMAP.md`. Read both before large changes.

## Build & test

- Rust is installed for the current user (GNU host toolchain, no MSVC on this machine).
  If `cargo` is not on PATH in a shell, prepend: `$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"`.
- Core tests: `cd core; cargo test` (pure Rust, no browser needed).
- wasm target installed: `wasm32-unknown-unknown`; `wasm-pack` lives in `~\.cargo\bin`.
- `npm run build:wasm` → builds `packages/tracear/src/wasm/` (gitignored build artifact;
  required after a fresh clone and after any core change).
- `npm run build:sdk` → dist build of the npm package (tsc + wasm copy + size budget).
  Dev servers consume `src/` via the `development` exports condition; `vite build`
  consumes `dist/` — rebuild the SDK before judging a production build.
- `npm run demo` → HTTPS dev server (self-signed, for phone testing over LAN).
  For local desktop work set `NO_SSL=1` first — localhost is a secure context on
  plain HTTP, so no cert warning.
- Demo has a "Self test (no camera)" button: full worker+WASM detection on a
  synthetic scene — use it to verify the pipeline without a camera.

## Conventions (do not break)

- Pixel centers at integer coordinates; y grows downward.
- Homographies are 3×3 row-major, map **marker level-0 px → frame level-0 px** unless a
  variable name says otherwise (`h_dst_to_src` etc.). Applied as p' = H·(x,y,1), then divide by w.
- Keypoint coordinates inside per-level code are in that pyramid level's pixel space;
  anything crossing module boundaries is converted to level-0 coordinates.
- Determinism matters: all randomness (RANSAC sampling, BRIEF pattern, synthetic data)
  goes through the seeded `rng::XorShift64`. Never use system RNG/time in the core.
- Scalar Rust first, correctness proven by tests; SIMD optimization comes later (M4)
  and must not change results beyond documented tolerances.
