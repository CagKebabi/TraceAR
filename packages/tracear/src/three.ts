/**
 * three.js adapter: anchors a THREE.Group to each tracked target and drives
 * a camera whose projection matches the (self-calibrated) phone camera.
 *
 * Usage:
 *   const t3 = new TracearThree(tracker);
 *   scene.add(t3.anchor(0));            // put content inside the anchor
 *   // per render frame:
 *   t3.update();                         // render-time predicted poses
 *   renderer.render(scene, t3.camera);
 *
 * Anchor space: origin at the marker center, X right, Y up, Z out of the
 * marker toward the viewer; 1 unit = the target's configured physical width
 * unit. The three camera sits at the origin; anchors move.
 */
import * as THREE from "three";
import type { CameraIntrinsics, Tracear } from "./index";

export interface TracearThreeOptions {
  near?: number;
  far?: number;
}

export class TracearThree {
  /** Camera with a projection matrix derived from the tracker's intrinsics. */
  readonly camera: THREE.PerspectiveCamera;
  private tracker: Tracear;
  private anchors = new Map<number, THREE.Group>();
  private near: number;
  private far: number;

  constructor(tracker: Tracear, opts: TracearThreeOptions = {}) {
    this.tracker = tracker;
    this.near = opts.near ?? 0.01;
    this.far = opts.far ?? 100;
    this.camera = new THREE.PerspectiveCamera();
    this.camera.matrixAutoUpdate = false;
    tracker.on("targetFound", ({ index }) => {
      const a = this.anchors.get(index);
      if (a) a.visible = true;
    });
    tracker.on("targetLost", ({ index }) => {
      const a = this.anchors.get(index);
      if (a) a.visible = false;
    });
  }

  /** Group that follows target `index`; add it to your scene once. */
  anchor(index: number): THREE.Group {
    let g = this.anchors.get(index);
    if (!g) {
      g = new THREE.Group();
      g.matrixAutoUpdate = false;
      g.visible = false;
      this.anchors.set(index, g);
    }
    return g;
  }

  /** Call once per render frame (before renderer.render). */
  update(timeMs: number = performance.now()): void {
    const intr = this.tracker.intrinsics();
    if (intr) this.setProjection(intr);
    for (const [index, g] of this.anchors) {
      const m = this.tracker.poseAt(index, timeMs);
      if (!m) continue;
      // OpenCV camera (x right, y down, z forward) -> three camera
      // (x right, y up, z backward): negate rows 1 and 2.
      const e = g.matrix.elements;
      for (let c = 0; c < 4; c++) {
        e[c * 4] = m[c * 4];
        e[c * 4 + 1] = -m[c * 4 + 1];
        e[c * 4 + 2] = -m[c * 4 + 2];
        e[c * 4 + 3] = m[c * 4 + 3];
      }
      g.matrixWorldNeedsUpdate = true;
    }
  }

  private setProjection(intr: CameraIntrinsics): void {
    const { fx, fy, cx, cy, width: w, height: h } = intr;
    const n = this.near;
    const f = this.far;
    // Pinhole -> WebGL projection for y-down pixel coordinates (see docs).
    const p = this.camera.projectionMatrix.elements; // column-major
    p.fill(0);
    p[0] = (2 * fx) / w;
    p[5] = (2 * fy) / h;
    p[8] = 1 - (2 * cx) / w;
    p[9] = (2 * cy) / h - 1;
    p[10] = -(f + n) / (f - n);
    p[11] = -1;
    p[14] = (-2 * f * n) / (f - n);
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }
}
