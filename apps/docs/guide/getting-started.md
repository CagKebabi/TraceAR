# Getting started

TraceAR tracks a known image (a poster, a card, packaging…) with the phone
camera, in the browser, and gives you a filtered 6DoF pose you can render
content on. Everything runs on-device; nothing is uploaded anywhere.

## Install

```sh
npm i @tracear/sdk
```

three.js is an optional peer dependency — install it if you use the
[`TracearThree` adapter](/guide/three):

```sh
npm i three
```

::: warning Vite users
Exclude the SDK from dev-server prebundling, or the worker and WASM URLs
break:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ["@tracear/sdk"] },
});
```
:::

## 1 · Compile a marker

Targets are compiled ahead of time into a `.tracear` file:

```sh
npx tracear compile poster.png   # writes poster.tracear
```

Serve the file with your app's static assets. (You can also compile
[in the browser](/reference/compiler), e.g. for user-uploaded images.)

## 2 · Track

```ts
import { Tracear } from "@tracear/sdk";

const tracker = await Tracear.create({
  container: document.querySelector("#ar")!, // gets the managed <video>
  targets: ["/poster.tracear"],
});

tracker.on("targetFound", ({ index }) => console.log("found", index));
tracker.on("targetLost", ({ index }) => console.log("lost", index));
tracker.on("update", (e) => {
  // e.homography, e.pose, e.quality … — see Tracking & poses
});

await tracker.start(); // asks for camera permission
```

`create()` loads the targets and boots the tracking worker; `start()` opens
the camera and begins processing. When you're done: `stop()` releases the
camera, `dispose()` also terminates the worker and removes the video element.

## 3 · Render something

The fastest path to 3D content is the three.js adapter:

```ts
import { TracearThree } from "@tracear/sdk/three";

const t3 = new TracearThree(tracker);
scene.add(t3.anchor(0)); // put your content inside this group

renderer.setAnimationLoop(() => {
  t3.update();                    // render-time predicted poses
  renderer.render(scene, t3.camera);
});
```

The full recipe — canvas overlay setup, marker-sized content, teardown — is
in [Rendering with three.js](/guide/three).

## Requirements

- **HTTPS** (or `localhost`): the camera API only exists in secure contexts.
- A modern mobile or desktop browser — see
  [Performance & browsers](/guide/performance) for the support matrix.

## Try it without writing code

The [live demo](https://cagkebabi.github.io/TraceAR/demo/) compiles a marker
in your browser and tracks it with your camera — including a camera-free
self test of the full pipeline.
