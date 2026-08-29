# Marker compiler & CLI

Targets are compiled from ordinary images into `.tracear` files: detection
features plus sub-pixel tracking patches, with the image downscaled so its
long side is at most 512 px (plenty for detection; keeps files small).

## CLI

```sh
npx tracear compile <image.png|image.jpg> [output.tracear]
npx tracear compile <image...> -o <output.tracear>    # multi-marker pack
npx tracear pack <file.tracear...> -o <output.tracear>
```

```sh
$ npx tracear compile poster.png
  poster.png: 168.9 KB, 1362 detection features (512x512)
poster.tracear written

$ npx tracear compile a.png b.png c.png -o album.tracear
album.tracear: 512.4 KB, 3 markers packed
```

PNG and JPEG input. With no output path, the extension is replaced with
`.tracear`. The reported feature count is your marker-quality signal — see
[Markers](/guide/markers#what-makes-a-good-marker). `pack` bundles existing
single-marker files without recompiling anything.

## `compileImage`

Browser-side compilation (e.g. user-uploaded images). It runs the same WASM
compiler on the calling thread:

```ts
import { compileImage } from "@tracear/sdk/compiler";

const result = await compileImage(source, maxSize?);
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | `ImageBitmapSource` | — | `File`, `Blob`, `<img>`, `<canvas>`, `ImageData`… |
| `maxSize` | `number` | `512` | Long-side cap before compilation. |

### `CompileResult`

| Field | Type | Description |
|---|---|---|
| `data` | `Uint8Array` | `.tracear` bytes — feed directly to `TracearConfig.targets`, or save as a file. |
| `featureCount` | `number` | Detection features found. |
| `width` / `height` | `number` | Size the image was compiled at (marker px). |

```ts
// Compile an upload and start tracking it immediately
const { data, featureCount } = await compileImage(fileInput.files[0]);
if (featureCount < 150) console.warn("weak marker — pick a busier image");
const tracker = await Tracear.create({ container, targets: [data] });
```

## `packMarkers` {#packmarkers}

Bundle compiled single-marker blobs into one multi-marker pack — pure byte
concatenation, nothing is recompiled:

```ts
import { compileImage, packMarkers } from "@tracear/sdk/compiler";

const a = await compileImage(photoA);
const b = await compileImage(photoB);
const album = packMarkers([a.data, b.data]); // Uint8Array — one .tracear pack

await Tracear.create({ container, targets: [album] }); // indices 0 and 1
```

| Parameter | Type | Description |
|---|---|---|
| `markers` | `(Uint8Array \| ArrayBuffer)[]` | Single-marker `.tracear` blobs, in the index order you want. |

::: tip
Compilation is fast (~0.1 s for a 512 px image) but not free — for fixed
targets, compile once with the CLI and ship the `.tracear` file instead of
compiling on every page load.
:::
