# Rendering with three.js

`TracearThree` (exported from `@tracear/sdk/three`) does two jobs:

1. Drives a `THREE.PerspectiveCamera` whose projection matches the phone
   camera's self-calibrated intrinsics — so perspective is correct and
   content sits *on* the marker, not floating near it.
2. Gives you one `THREE.Group` per target ("anchor") that follows the
   render-time predicted pose and toggles visibility on found/lost.

## Full recipe

```ts
import * as THREE from "three";
import { Tracear } from "@tracear/sdk";
import { TracearThree } from "@tracear/sdk/three";

const container = document.querySelector<HTMLElement>("#ar")!;
container.style.position = "relative"; // video + canvas stack inside

const tracker = await Tracear.create({
  container,
  targets: ["/poster.tracear"],
  targetWidthsMeters: [0.21],
});

// Transparent canvas over the managed <video>
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
Object.assign(renderer.domElement.style, {
  position: "absolute",
  inset: "0",
  width: "100%",
  height: "100%",
  pointerEvents: "none",
});
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 1.2));

const t3 = new TracearThree(tracker);
const anchor = t3.anchor(0);
scene.add(anchor);

// Content: a box sitting ON the marker (marker plane is z = 0, +Z toward viewer)
const box = new THREE.Mesh(
  new THREE.BoxGeometry(0.1, 0.1, 0.1),
  new THREE.MeshStandardMaterial({ color: 0x4945ff }),
);
box.position.z = 0.05;
anchor.add(box);

await tracker.start();

function fitCanvas() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h, false);
}
new ResizeObserver(fitCanvas).observe(container);
fitCanvas();

renderer.setAnimationLoop(() => {
  t3.update(); // must run every frame, before render
  renderer.render(scene, t3.camera);
});
```

## Anchor space

- Origin at the **marker center**; X right, Y up, Z out of the marker toward
  the viewer.
- 1 unit = the unit you used in `targetWidthsMeters`. With `0.21` (meters), a
  0.21-wide plane exactly covers the marker.
- The camera stays at the scene origin; anchors move. Don't reposition
  `t3.camera` yourself — its matrix is driven by the tracker.

A plane that exactly covers the marker (e.g. for video-on-marker effects):

```ts
const cover = new THREE.Mesh(
  new THREE.PlaneGeometry(0.21, 0.21 * (markerHeight / markerWidth)),
  material,
);
anchor.add(cover); // marker plane is z = 0 — no offset needed
```

`markerWidth` / `markerHeight` (pixel size of the compiled marker) come from
any `update` event.

## Loading a GLTF model

```ts
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const gltf = await new GLTFLoader().loadAsync("/model.glb");
gltf.scene.scale.setScalar(0.1);
anchor.add(gltf.scene);
```

## Teardown

Mobile browsers keep pages alive in the back-forward cache; release the
camera and GPU resources when the page hides:

```ts
window.addEventListener("pagehide", () => {
  tracker.dispose();
  renderer.dispose();
  renderer.forceContextLoss();
});
```

## Why `t3.update()` instead of the update event?

Pose measurements arrive at camera rate (~25–30 Hz) while the display runs at
60 Hz+. `t3.update()` calls [`tracker.poseAt()`](/reference/tracear#poseat)
with the render timestamp, which extrapolates over the pipeline latency and
smooths orientation per render frame — that is where the "glued to the
marker" feel comes from. Rendering straight from `update` events would
visibly step.
