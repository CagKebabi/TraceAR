import * as THREE from "three";
import { Tracear, type UpdateEvent } from "tracear";
import { compileImage } from "tracear/compiler";
import { TracearThree } from "tracear/three";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const btnSample = $<HTMLButtonElement>("btn-sample");
const fileInput = $<HTMLInputElement>("file-input");
const markerStatus = $<HTMLParagraphElement>("marker-status");
const markerPanel = $<HTMLDivElement>("marker-panel");
const markerPreview = $<HTMLCanvasElement>("marker-preview");
const downloadLink = $<HTMLAnchorElement>("download-link");
const btnStart = $<HTMLButtonElement>("btn-start");
const btnSelfTest = $<HTMLButtonElement>("btn-selftest");
const stats = $<HTMLSpanElement>("stats");
const container = $<HTMLDivElement>("ar-container");
const overlay = $<HTMLCanvasElement>("overlay");

let markerBytes: Uint8Array | null = null;
let markerSource: CanvasImageSource | null = null;
let tracker: Tracear | null = null;
let clearTimer: number | null = null;
let three: { renderer: THREE.WebGLRenderer; scene: THREE.Scene; t3: TracearThree } | null = null;
let threeGen = 0; // invalidates stale rVFC/rAF loops across restarts

/** Transparent WebGL layer over the video: a cube + axes anchored to the marker. */
function setupThree(t: Tracear): void {
  if (three) {
    three.renderer.dispose();
    // Release the WebGL context immediately — Safari caps live contexts per
    // tab and restarts otherwise accumulate zombies until everything freezes.
    try {
      three.renderer.forceContextLoss();
    } catch {
      /* already lost */
    }
    three.renderer.domElement.remove();
  }
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.className = "overlay3d";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(0.5, 1, 1);
  scene.add(dir);

  const t3 = new TracearThree(t);
  const anchor = t3.anchor(0);
  // Marker width = 1 unit; a cube sitting on the marker center.
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.3, 0.3),
    new THREE.MeshStandardMaterial({ color: 0x2a6df4, roughness: 0.35, metalness: 0.1 }),
  );
  cube.position.z = 0.15; // object frame: Z points out of the marker
  anchor.add(cube);
  anchor.add(new THREE.AxesHelper(0.5));
  // Diagnostic plane hugging the marker: if this stays glued while the cube
  // top drifts, the focal estimate is off; if both swim, it's latency/pose.
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ color: 0x39d98a, transparent: true, opacity: 0.15, side: THREE.DoubleSide }),
  );
  anchor.add(plane);
  scene.add(anchor);

  three = { renderer, scene, t3 };
  const gen = ++threeGen;

  // Poses advance ONLY when a new camera frame is displayed: updating the
  // anchor at render rate would make the cube glide while the video steps at
  // camera rate — that mismatch reads as swimming.
  const video = t.video;
  const onVideoFrame = () => {
    if (!three || gen !== threeGen) return;
    three.t3.update(performance.now());
    video.requestVideoFrameCallback(onVideoFrame);
  };
  if ("requestVideoFrameCallback" in video) {
    video.requestVideoFrameCallback(onVideoFrame);
  }

  const loop = () => {
    if (!three || gen !== threeGen) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    const canvas = three.renderer.domElement;
    if (canvas.width !== Math.round(w * devicePixelRatio) || canvas.height !== Math.round(h * devicePixelRatio)) {
      three.renderer.setSize(w, h, false);
    }
    if (!("requestVideoFrameCallback" in video)) {
      three.t3.update(performance.now()); // rAF fallback
    }
    three.renderer.render(three.scene, three.t3.camera);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

/** Deterministic corner-rich pattern — same idea as the core's synthetic texture. */
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
  // White quiet border helps printing and framing.
  ctx.strokeStyle = "white";
  ctx.lineWidth = 16;
  ctx.strokeRect(8, 8, size - 16, size - 16);
  return c;
}

async function setMarker(source: CanvasImageSource & ImageBitmapSource, label: string) {
  markerStatus.textContent = "Compiling marker…";
  try {
    const res = await compileImage(source);
    markerBytes = res.data;
    markerSource = source;
    tracker?.dispose();
    tracker = null;
    markerStatus.textContent = `${label}: ${res.featureCount} features, ${(res.data.length / 1024).toFixed(1)} KB (${res.width}x${res.height})`;
    // Offer the compiled target for download — the file apps ship to
    // Tracear.create({ targets }).
    const tracearBlob = new Blob([res.data.slice()], { type: "application/octet-stream" });
    const dl = document.getElementById("download-tracear") as HTMLAnchorElement | null;
    if (dl) {
      if (dl.href) URL.revokeObjectURL(dl.href);
      dl.href = URL.createObjectURL(tracearBlob);
      dl.hidden = false;
    }
    const pctx = markerPreview.getContext("2d")!;
    pctx.clearRect(0, 0, markerPreview.width, markerPreview.height);
    pctx.drawImage(source, 0, 0, markerPreview.width, markerPreview.height);
    if (source instanceof HTMLCanvasElement) {
      downloadLink.href = source.toDataURL("image/png");
      downloadLink.hidden = false;
    } else {
      downloadLink.hidden = true;
    }
    markerPanel.hidden = false;
    btnStart.disabled = false;
    btnSelfTest.disabled = false;
  } catch (e) {
    markerStatus.textContent = `Compile failed: ${e instanceof Error ? e.message : e}`;
  }
}

btnSample.onclick = () => setMarker(drawSampleMarker(), "Sample marker");

fileInput.onchange = async () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  const bmp = await createImageBitmap(f);
  await setMarker(bmp, f.name);
};

function applyH(h: Float64Array, x: number, y: number): [number, number] {
  const w = h[6] * x + h[7] * y + h[8];
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

const msAvg: number[] = [];

function drawQuad(e: UpdateEvent) {
  overlay.width = e.processWidth;
  overlay.height = e.processHeight;
  const ctx = overlay.getContext("2d")!;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const mw = e.markerWidth;
  const mh = e.markerHeight;
  const corners = [
    applyH(e.homography, 0, 0),
    applyH(e.homography, mw, 0),
    applyH(e.homography, mw, mh),
    applyH(e.homography, 0, mh),
  ];
  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
  ctx.closePath();
  ctx.strokeStyle = "#39d98a";
  ctx.lineWidth = 3;
  ctx.stroke();
  // Top-edge accent shows orientation.
  ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]);
  ctx.lineTo(corners[1][0], corners[1][1]);
  ctx.strokeStyle = "#ff5c7a";
  ctx.stroke();

  msAvg.push(e.workerMs);
  if (msAvg.length > 30) msAvg.shift();
  const avg = msAvg.reduce((a, b) => a + b, 0) / msAvg.length;
  const mode = e.tracking ? "track" : "detect";
  const intr = tracker?.intrinsics();
  const focal = intr ? ` · f ${(intr.fx / intr.width).toFixed(2)}` : "";
  stats.textContent = `${mode} ${avg.toFixed(1)} ms · ${e.inliers}/${e.matches} · q ${e.quality.toFixed(2)}${focal}`;

  if (clearTimer !== null) clearTimeout(clearTimer);
  clearTimer = window.setTimeout(() => {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }, 400);
}

/**
 * End-to-end pipeline check without a camera: render the marker into a fake
 * "camera frame" (rotated + scaled on a noisy background), run one-shot
 * detection through the same worker the live path uses, and overlay the
 * result. Marker center is placed at a known point so the result can be
 * sanity-checked numerically, too.
 */
btnSelfTest.onclick = async () => {
  if (!markerBytes) return;
  btnSelfTest.disabled = true;
  stats.textContent = "self test running…";
  try {
    const scene = document.createElement("canvas");
    scene.width = 640;
    scene.height = 480;
    const ctx = scene.getContext("2d")!;
    // noisy background
    ctx.fillStyle = "#666";
    ctx.fillRect(0, 0, 640, 480);
    for (let i = 0; i < 300; i++) {
      const v = Math.floor(Math.random() * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(Math.random() * 620, Math.random() * 460, 4 + Math.random() * 24, 4 + Math.random() * 24);
    }
    // marker at center, rotated 25deg, ~280px on screen
    ctx.save();
    ctx.translate(320, 240);
    ctx.rotate((25 * Math.PI) / 180);
    ctx.drawImage(markerSource!, -140, -140, 280, 280);
    ctx.restore();

    const t = tracker ?? (tracker = await Tracear.create({ container, targets: [markerBytes] }));
    const results = await t.detectImage(scene);
    const r = results[0];
    if (!r) {
      stats.textContent = "self test: marker NOT found";
      return;
    }
    // center of marker should land near scene center (scaled to process size)
    const [cx, cy] = applyH(r.homography, r.markerWidth / 2, r.markerHeight / 2);
    const sx = r.processWidth / 640;
    const err = Math.hypot(cx - 320 * sx, cy - 240 * sx);
    drawQuad(r);
    stats.textContent =
      `self test OK · ${r.workerMs.toFixed(1)} ms · inliers ${r.inliers}/${r.matches} · center err ${err.toFixed(1)} px`;
  } catch (e) {
    stats.textContent = `self test failed: ${e instanceof Error ? e.message : e}`;
  } finally {
    btnSelfTest.disabled = false;
  }
};

btnStart.onclick = async () => {
  if (!markerBytes) return;
  btnStart.disabled = true;
  btnStart.textContent = "Starting…";
  try {
    tracker = tracker ?? (await Tracear.create({ container, targets: [markerBytes] }));
    tracker.on("update", drawQuad);
    tracker.on("targetFound", () => (stats.textContent = "target found"));
    tracker.on("targetLost", () => (stats.textContent = "target lost — searching…"));
    tracker.on("error", ({ message }) => (stats.textContent = `error: ${message}`));
    setupThree(tracker);
    await tracker.start();
    btnStart.textContent = "Camera running";
    stats.textContent = "searching for marker…";
  } catch (e) {
    btnStart.disabled = false;
    btnStart.textContent = "Start camera";
    stats.textContent = `camera failed: ${e instanceof Error ? e.message : e}`;
  }
};
