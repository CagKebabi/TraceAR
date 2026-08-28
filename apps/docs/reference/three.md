# `TracearThree`

The three.js adapter. Requires `three >= 0.150` (optional peer dependency).

```ts
import { TracearThree } from "@tracear/sdk/three";

const t3 = new TracearThree(tracker, { near: 0.01, far: 100 });
```

Worked example with renderer setup: [Rendering with three.js](/guide/three).

## `new TracearThree(tracker, options?)`

| Option | Type | Default | Description |
|---|---|---|---|
| `near` | `number` | `0.01` | Camera near plane (in your pose units). |
| `far` | `number` | `100` | Camera far plane. |

## Properties

### `camera`

```ts
readonly camera: THREE.PerspectiveCamera
```

A camera whose projection matrix is derived from the tracker's
self-calibrated [intrinsics](/reference/tracear#intrinsics) every frame.
Render with it; don't move it — the camera stays at the origin and anchors
move.

## Methods

### `anchor(index)`

```ts
anchor(index: number): THREE.Group
```

The group that follows target `index`. Add it to your scene once and put
your content inside. It becomes visible on `targetFound` and hides on
`targetLost`.

Anchor space: origin at the marker center, X right, Y up, Z out of the
marker toward the viewer; 1 unit = the target's configured physical width
unit.

### `update(timeMs?)`

```ts
update(timeMs: number = performance.now()): void
```

Call once per render frame, before `renderer.render(scene, t3.camera)`.
Pulls the render-time predicted pose for every anchor
([`poseAt`](/reference/tracear#poseat)), converts OpenCV → WebGL axes, and
refreshes the camera projection.
