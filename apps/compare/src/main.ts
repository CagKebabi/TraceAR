import * as THREE from "three";
import { Tracear, type UpdateEvent } from "@tracear/sdk";
import { compileImage } from "@tracear/sdk/compiler";
import { TracearThree } from "@tracear/sdk/three";

/**
 * MindAR's compiler yields via tf.nextFrame(), and tfjs captures the global
 * requestAnimationFrame AT MODULE LOAD — so this wrapper must be installed
 * before mind-ar/tfjs are (dynamically) imported. Browsers pause rAF entirely
 * in hidden tabs, which would freeze the compile if the user switches away;
 * while a compile is running in a hidden tab, rAF is routed through a
 * MessageChannel (message tasks are not throttled). Everywhere else the real
 * rAF is used, so render loops keep normal browser behavior.
 */
let compileBoost = false;
{
  const realRaf = window.requestAnimationFrame.bind(window);
  const chan = new MessageChannel();
  const pending: FrameRequestCallback[] = [];
  chan.port1.onmessage = () => pending.shift()?.(performance.now());
  window.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    if (compileBoost && document.visibilityState === "hidden") {
      pending.push(cb);
      chan.port2.postMessage(0);
      return 0;
    }
    return realRaf(cb);
  }) as typeof window.requestAnimationFrame;
}

type EngineName = "tracear" | "mindar";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

// Phone debugging without devtools: surface any uncaught error on the page.
window.addEventListener("error", (e) => {
  const live = document.getElementById("live");
  if (live) live.textContent = `error: ${e.message}`;
});
window.addEventListener("unhandledrejection", (e) => {
  const live = document.getElementById("live");
  if (live) live.textContent = `error: ${e.reason instanceof Error ? e.reason.message : e.reason}`;
});
const btnSample = $<HTMLButtonElement>("btn-sample");
const fileInput = $<HTMLInputElement>("file-input");
const markerStatus = $<HTMLParagraphElement>("marker-status");
const markerPanel = $<HTMLDivElement>("marker-panel");
const markerPreview = $<HTMLCanvasElement>("marker-preview");
const downloadLink = $<HTMLAnchorElement>("download-link");
const engTracear = $<HTMLButtonElement>("eng-tracear");
const engMindar = $<HTMLButtonElement>("eng-mindar");
const btnStart = $<HTMLButtonElement>("btn-start");
const live = $<HTMLParagraphElement>("live");
const view = $<HTMLDivElement>("view");
const btnReset = $<HTMLButtonElement>("btn-reset");
const resultsBody = document.querySelector<HTMLTableSectionElement>("#results tbody")!;

/* ------------------------------ metrics ------------------------------- */

/**
 * High-frequency jitter of recent (x, y) samples, in 640px-frame units.
 * Uses the RMS of second differences (curvature), which cancels constant
 * velocity — so deliberate camera motion does not read as "jitter", only
 * frame-to-frame noise does. For white noise E[|d2|^2] = 1.5 sigma^2 per
 * axis; dividing by 1.5 makes the value an unbiased per-frame sigma.
 */
class JitterMeter {
  private buf: [number, number][] = [];
  private lastPush = 0;
  push(x: number, y: number): void {
    // A gap (target lost / engine re-acquiring) breaks sample continuity —
    // the jump would read as a huge second difference. Start fresh instead.
    const now = performance.now();
    if (now - this.lastPush > 250) this.buf = [];
    this.lastPush = now;
    this.buf.push([x, y]);
    if (this.buf.length > 90) this.buf.shift();
  }
  clear(): void {
    this.buf = [];
  }
  rms(): number | null {
    const n = this.buf.length;
    if (n < 30) return null;
    let s = 0;
    let cnt = 0;
    for (let i = 1; i + 1 < n; i++) {
      const ax = this.buf[i][0] - (this.buf[i - 1][0] + this.buf[i + 1][0]) / 2;
      const ay = this.buf[i][1] - (this.buf[i - 1][1] + this.buf[i + 1][1]) / 2;
      s += ax * ax + ay * ay;
      cnt++;
    }
    return Math.sqrt(s / cnt / 1.5);
  }
}

class EngineStats {
  jitterSnapshots: number[] = [];
  updateCount = 0;
  runSeconds = 0;
  workerMs: number[] = [];
  reset(): void {
    this.jitterSnapshots = [];
    this.updateCount = 0;
    this.runSeconds = 0;
    this.workerMs = [];
  }
  private pct(v: number[], p: number): number | null {
    if (!v.length) return null;
    const s = [...v].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
  }
  summary() {
    return {
      medianJitter: this.pct(this.jitterSnapshots, 50),
      p90Jitter: this.pct(this.jitterSnapshots, 90),
      updatesPerSec: this.runSeconds > 1 ? this.updateCount / this.runSeconds : null,
      avgWorkerMs: this.workerMs.length
        ? this.workerMs.reduce((a, b) => a + b, 0) / this.workerMs.length
        : null,
    };
  }
}

const stats: Record<EngineName, EngineStats> = { tracear: new EngineStats(), mindar: new EngineStats() };

function renderResults(): void {
  const t = stats.tracear.summary();
  const m = stats.mindar.summary();
  const fmt = (v: number | null, digits = 2, unit = "") => (v === null ? "—" : v.toFixed(digits) + unit);
  const row = (
    label: string,
    tv: number | null,
    mv: number | null,
    lowerWins: boolean,
    digits: number,
    unit: string,
  ) => {
    let tw = "";
    let mw = "";
    if (tv !== null && mv !== null && tv !== mv) {
      const tracearWins = lowerWins ? tv < mv : tv > mv;
      tw = tracearWins ? " class=\"win\"" : "";
      mw = tracearWins ? "" : " class=\"win\"";
    }
    return `<tr><td>${label}</td><td${tw}>${fmt(tv, digits, unit)}</td><td${mw}>${fmt(mv, digits, unit)}</td></tr>`;
  };
  resultsBody.innerHTML =
    row("median jitter (px)", t.medianJitter, m.medianJitter, true, 3, "") +
    row("p90 jitter (px)", t.p90Jitter, m.p90Jitter, true, 3, "") +
    row("pose updates / s", t.updatesPerSec, m.updatesPerSec, false, 1, "") +
    `<tr><td>CV time / frame</td><td>${fmt(t.avgWorkerMs, 1, " ms")}</td><td>n/a (not exposed)</td></tr>`;
}

/* ------------------------------ marker -------------------------------- */

function drawSampleMarker(size = 512): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d")!;
  let seed = 0x12345678;
  const rnd = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return ((seed >>> 0) % 100000) / 100000;
  };
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, size, size);
  for (let i = 0; i < 240; i++) {
    const w = 14 + rnd() * 60;
    const h = 14 + rnd() * 60;
    const x = rnd() * (size - w);
    const y = rnd() * (size - h);
    const v = Math.floor(35 + rnd() * 195);
    ctx.fillStyle = `rgb(${v},${v},${v})`;
    ctx.fillRect(x, y, w, h);
  }
  ctx.strokeStyle = "white";
  ctx.lineWidth = 16;
  ctx.strokeRect(8, 8, size - 16, size - 16);
  return c;
}

let tracearBytes: Uint8Array | null = null;
let mindUrl: string | null = null;

async function prepareMarker(canvas: HTMLCanvasElement, label: string): Promise<void> {
  btnStart.disabled = true;
  tracearBytes = null;
  mindUrl = null;

  const pctx = markerPreview.getContext("2d")!;
  pctx.drawImage(canvas, 0, 0, markerPreview.width, markerPreview.height);
  downloadLink.href = canvas.toDataURL("image/png");
  markerPanel.hidden = false;

  markerStatus.textContent = `${label}: compiling for Tracear…`;
  const t0 = performance.now();
  const res = await compileImage(canvas);
  tracearBytes = res.data;
  const tracearMs = performance.now() - t0;

  markerStatus.textContent = `${label}: compiling for MindAR (their compiler is slow — hang on)…`;
  const { Compiler } = await import("mind-ar/dist/mindar-image.prod.js");
  const img = new Image();
  img.src = canvas.toDataURL("image/png");
  await new Promise((ok, err) => {
    img.onload = ok;
    img.onerror = err;
  });
  const t1 = performance.now();
  const compiler = new Compiler();
  let buffer: ArrayBuffer;
  compileBoost = true; // see the rAF wrapper at the top of this file
  try {
    await compiler.compileImageTargets([img], (p: number) => {
      markerStatus.textContent = `${label}: compiling for MindAR… ${p.toFixed(0)}%`;
    });
    buffer = await compiler.exportData();
  } finally {
    compileBoost = false;
  }
  const mindMs = performance.now() - t1;
  if (mindUrl) URL.revokeObjectURL(mindUrl);
  mindUrl = URL.createObjectURL(new Blob([buffer]));

  markerStatus.textContent =
    `${label}: ready · Tracear compile ${(tracearMs / 1000).toFixed(1)}s (${(res.data.length / 1024).toFixed(0)} KB) · ` +
    `MindAR compile ${(mindMs / 1000).toFixed(1)}s (${(buffer.byteLength / 1024).toFixed(0)} KB)`;
  btnStart.disabled = false;
}

btnSample.onclick = () => prepareMarker(drawSampleMarker(), "Sample marker").catch(showError);
fileInput.onchange = async () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  const bmp = await createImageBitmap(f);
  const c = document.createElement("canvas");
  const scale = Math.min(1, 512 / Math.max(bmp.width, bmp.height));
  c.width = Math.round(bmp.width * scale);
  c.height = Math.round(bmp.height * scale);
  c.getContext("2d")!.drawImage(bmp, 0, 0, c.width, c.height);
  prepareMarker(c, f.name).catch(showError);
};

function showError(e: unknown): void {
  markerStatus.textContent = `error: ${e instanceof Error ? e.message : e}`;
}

/* ------------------------------ engines ------------------------------- */

function anchorContent(): THREE.Group {
  const g = new THREE.Group();
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.3, 0.3),
    new THREE.MeshStandardMaterial({ color: 0x2a6df4, roughness: 0.35, metalness: 0.1 }),
  );
  cube.position.z = 0.15;
  g.add(cube);
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0x39d98a, transparent: true, opacity: 0.15, side: THREE.DoubleSide }),
  );
  g.add(plane);
  return g;
}

/**
 * Release the WebGL context NOW, not at garbage collection: Safari caps live
 * contexts per tab (~8-16), and engine restarts otherwise accumulate zombie
 * contexts until the browser force-drops them and everything starts freezing.
 */
function killRenderer(renderer: THREE.WebGLRenderer): void {
  try {
    renderer.dispose();
    renderer.forceContextLoss();
  } catch {
    /* already lost */
  }
}

function addLights(scene: THREE.Scene): void {
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(0.5, 1, 1);
  scene.add(dir);
}

interface RunningEngine {
  name: EngineName;
  stop(): Promise<void> | void;
}

let running: RunningEngine | null = null;
let selected: EngineName = "tracear";
const jitter = new JitterMeter();
let statsTimer: number | null = null;
let runStart = 0;
/** Sticky diagnostic message; suppresses the periodic stats line. */
let warnMsg: string | null = null;

function beginStatsLoop(name: EngineName, extra: () => string): void {
  runStart = performance.now();
  jitter.clear();
  const baseSeconds = stats[name].runSeconds; // accumulate across restarts
  statsTimer = window.setInterval(() => {
    const st = stats[name];
    st.runSeconds = baseSeconds + (performance.now() - runStart) / 1000;
    const r = jitter.rms();
    if (r !== null) st.jitterSnapshots.push(r);
    if (warnMsg) {
      live.textContent = warnMsg;
      renderResults();
      return;
    }
    live.textContent =
      `${name.toUpperCase()} · jitter ${r === null ? "…" : r.toFixed(3) + " px"} · ` +
      `${st.runSeconds > 1 ? (st.updateCount / st.runSeconds).toFixed(1) : "…"} updates/s${extra()}`;
    renderResults();
  }, 500);
}

function endStatsLoop(): void {
  if (statsTimer !== null) clearInterval(statsTimer);
  statsTimer = null;
}

async function startTracear(): Promise<RunningEngine> {
  const tracker = await Tracear.create({ container: view, targets: [tracearBytes!.slice()] });
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.className = "overlay3d";
  view.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  addLights(scene);
  const t3 = new TracearThree(tracker);
  const anchor = t3.anchor(0);
  anchor.add(anchorContent());
  scene.add(anchor);

  const msWindow: number[] = [];
  let mode = "…";
  tracker.on("update", (e: UpdateEvent) => {
    stats.tracear.updateCount++;
    stats.tracear.workerMs.push(e.workerMs);
    msWindow.push(e.workerMs);
    if (msWindow.length > 30) msWindow.shift();
    mode = e.tracking ? "track" : "detect";
    // Marker center through the homography, normalized to a 640px frame.
    const h = e.homography;
    const [mx, my] = [e.markerWidth / 2, e.markerHeight / 2];
    const w = h[6] * mx + h[7] * my + h[8];
    const scale = 640 / e.processWidth;
    jitter.push(((h[0] * mx + h[1] * my + h[2]) / w) * scale, ((h[3] * mx + h[4] * my + h[5]) / w) * scale);
  });

  let raf = 0;
  let stopped = false;
  const video = tracker.video;
  const onVideoFrame = () => {
    if (stopped) return;
    t3.update(performance.now());
    video.requestVideoFrameCallback(onVideoFrame);
  };
  if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(onVideoFrame);
  const loop = () => {
    const w = view.clientWidth;
    const hpx = view.clientHeight;
    // Check BOTH dimensions: the container height grows when the video
    // arrives, and a stale small buffer stretches into a blurry cube.
    if (
      renderer.domElement.width !== Math.round(w * devicePixelRatio) ||
      renderer.domElement.height !== Math.round(hpx * devicePixelRatio)
    ) {
      renderer.setSize(w, hpx, false);
    }
    if (!("requestVideoFrameCallback" in video)) t3.update(performance.now());
    renderer.render(scene, t3.camera);
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  await tracker.start();
  beginStatsLoop("tracear", () => {
    const avg = msWindow.length ? msWindow.reduce((a, b) => a + b, 0) / msWindow.length : 0;
    return ` · ${mode} ${avg.toFixed(1)} ms`;
  });
  return {
    name: "tracear",
    stop() {
      stopped = true;
      cancelAnimationFrame(raf);
      endStatsLoop();
      tracker.dispose();
      killRenderer(renderer);
      view.innerHTML = "";
    },
  };
}

async function startMindar(): Promise<RunningEngine> {
  live.textContent = "loading MindAR engine…";
  const { MindARThree } = await import("mind-ar/dist/mindar-image-three.prod.js");
  // Give the container an explicit size (MindAR positions itself absolutely).
  view.style.height = `${Math.round(view.clientWidth * 0.75)}px`;
  const mindar = new MindARThree({
    container: view,
    imageTargetSrc: mindUrl!,
    uiScanning: "no",
    uiLoading: "no",
    uiError: "no",
  });
  const { renderer, scene, camera } = mindar;
  addLights(scene);
  const anchor = mindar.addAnchor(0);
  anchor.group.add(anchorContent());

  live.textContent = "starting MindAR camera — the FIRST start compiles tfjs shaders and can take 10-20 s on a black screen, hang on…";
  await mindar.start();
  // Match the container to the camera aspect so "cover" doesn't crop.
  const fitHeight = () => {
    if (mindar.video.videoWidth) {
      view.style.height = `${Math.round((view.clientWidth * mindar.video.videoHeight) / mindar.video.videoWidth)}px`;
    }
  };
  mindar.video.addEventListener("loadedmetadata", fitHeight);
  fitHeight();
  // Remote-debug watchdog: report what actually failed instead of a black box.
  const watchdog = window.setTimeout(() => {
    if (!mindar.video.videoWidth) {
      warnMsg =
        `MindAR: no camera frames after 10 s (video readyState ${mindar.video.readyState}, ` +
        `srcObject ${mindar.video.srcObject ? "set" : "missing"})`;
    }
  }, 10000);

  const pos = new THREE.Vector3();
  let lastX = NaN;
  let lastY = NaN;
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
    if (!anchor.group.visible || !mindar.video.videoWidth) return;
    pos.setFromMatrixPosition(anchor.group.matrixWorld);
    pos.project(camera);
    // NDC -> camera px -> 640px-frame px (same normalization as Tracear).
    const vw = mindar.video.videoWidth;
    const vh = mindar.video.videoHeight;
    const x = ((pos.x + 1) / 2) * vw * (640 / vw);
    const y = ((1 - pos.y) / 2) * vh * (640 / vw);
    if (x !== lastX || y !== lastY) {
      // count only real pose updates, not repeated render frames
      stats.mindar.updateCount++;
      jitter.push(x, y);
      lastX = x;
      lastY = y;
    }
  });

  beginStatsLoop("mindar", () => "");
  return {
    name: "mindar",
    stop() {
      clearTimeout(watchdog);
      endStatsLoop();
      renderer.setAnimationLoop(null);
      try {
        mindar.stop();
      } catch {
        /* mindar.stop throws if the camera never started */
      }
      killRenderer(renderer);
      view.innerHTML = "";
      view.style.height = "";
    },
  };
}

/* ------------------------------- wiring ------------------------------- */

function selectEngine(name: EngineName): void {
  selected = name;
  engTracear.classList.toggle("active", name === "tracear");
  engMindar.classList.toggle("active", name === "mindar");
  if (running && running.name !== name) {
    void restart();
  }
}

async function restart(): Promise<void> {
  btnStart.disabled = true;
  btnStart.textContent = "Starting…";
  warnMsg = null;
  if (running) {
    await running.stop();
    running = null;
  }
  try {
    running = selected === "tracear" ? await startTracear() : await startMindar();
    btnStart.textContent = "Restart camera";
  } catch (e) {
    live.textContent = `failed to start ${selected}: ${e instanceof Error ? e.message : e}`;
    btnStart.textContent = "Start camera";
  } finally {
    btnStart.disabled = false;
  }
}

engTracear.onclick = () => selectEngine("tracear");
engMindar.onclick = () => selectEngine("mindar");
btnStart.onclick = () => void restart();
btnReset.onclick = () => {
  stats.tracear.reset();
  stats.mindar.reset();
  renderResults();
};

renderResults();
