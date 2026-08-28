---
layout: home

hero:
  name: TraceAR
  text: Jitter-free image tracking for the mobile web
  tagline: A Rust → WASM AR engine with sub-pixel tracking and a self-calibrating camera — ~95 KB gzipped, free and MIT-licensed.
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Try the live demo
      link: https://cagkebabi.github.io/TraceAR/demo/
    - theme: alt
      text: GitHub
      link: https://github.com/CagKebabi/TraceAR

features:
  - icon: 🎯
    title: Sub-pixel tracking
    details: After detection, an inverse-compositional patch tracker locks on with sub-pixel precision — 1.90 px median jitter measured on a real phone.
  - icon: ⚡
    title: 2.5 ms per tracked frame
    details: Rust compiled to WASM with SIMD, running in a worker. The main thread only pumps frames and renders.
  - icon: 📷
    title: Self-calibrating pose
    details: Full 6DoF pose with online focal-length estimation. Your users never see a calibration step.
  - icon: 🧊
    title: three.js in five lines
    details: An adapter drives a correctly-projected camera and per-target anchor groups. Drop your content in and render.
  - icon: 📦
    title: ~95 KB gzipped
    details: WASM binary included. Two tiny image codecs are the only dependencies; three.js is an optional peer.
  - icon: 🆓
    title: Free & open source
    details: MIT-licensed, no keys, no accounts, no usage limits. Compile markers locally with one CLI command.
---
