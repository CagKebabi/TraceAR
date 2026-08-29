# Markers

A *marker* (or *target*) is the image TraceAR looks for. It is compiled once
into a compact `.tracear` file containing detection features and tracking
patches; the original image is not needed at runtime.

## What makes a good marker

The engine tracks visual texture, so the image itself decides how well
tracking works:

- **Rich, irregular detail** everywhere — photos, illustrations, busy poster
  art. Corners and blobs are features; empty space is not.
- **Contrast.** Tracking runs on luma; washed-out or very dark prints give
  weak features.
- **Avoid repetition and symmetry.** Tiled patterns and mirrored layouts
  create ambiguous matches.
- **Avoid large flat areas** — logos on white backgrounds are the classic
  hard case. Fill the frame with detail instead.
- **Matte beats glossy** when printing; glare wipes out texture under real
  lighting.

The compiler reports a feature count. As a rule of thumb, a few hundred
features track comfortably; if you get far fewer, pick a busier image.

## Compiling

### CLI

```sh
npx tracear compile poster.png            # → poster.tracear
npx tracear compile poster.jpg out.tracear
```

PNG and JPEG are supported. Images are downscaled so the long side is at most
512 px before compilation — plenty for detection, and it keeps files small
(the 512 px demo marker compiles to ~170 KB).

### In the browser

For user-uploaded images, compile client-side with
[`compileImage`](/reference/compiler):

```ts
import { compileImage } from "@tracear/sdk/compiler";

const { data, featureCount } = await compileImage(file); // File | Blob | img | canvas …
const tracker = await Tracear.create({ container, targets: [data] });
```

## Many targets, one file (packs)

A `.tracear` file can hold any number of markers. A *pack* is a pure
container: each entry is an unmodified single-marker file, so bundling is
byte concatenation — nothing gets recompiled when you add or remove one.

```sh
npx tracear compile a.png b.png c.png -o album.tracear   # compile + bundle
npx tracear pack a.tracear b.tracear -o album.tracear    # bundle existing files
```

In the browser, bundle compiled markers with `packMarkers` (see the
[compiler reference](/reference/compiler#packmarkers)).

Use a pack anywhere a target is accepted — it expands in place:

```ts
await Tracear.create({ container, targets: ["/album.tracear"] });
// markers get indices 0..N-1 in file order
```

Per-file targets and packs behave identically at runtime; choose whichever
fits your asset pipeline. Sessions with many targets stay cheap: frame
features are shared across all markers and idle-target detection is
amortized, so 10 targets cost roughly the same per frame as one — see
[Performance](/guide/performance#many-targets).

## Physical size

Poses come out in the unit you declare for the marker's width:

```ts
await Tracear.create({
  container,
  targets: ["/poster.tracear"],
  targetWidthsMeters: [0.21], // an A4 print is 21 cm wide
});
```

With the width set in meters, `pose.position` is in meters and a 1-unit
three.js box is 1 m tall. If you skip it, each target defaults to `1` — fine
whenever absolute scale doesn't matter.

## Showing the marker

- Print it, or display it on another screen — both work.
- Keep it flat. The pose model assumes a planar target; a curled print shows
  up as pose error.
- Bigger on screen = better. Tracking quality follows how many camera pixels
  the marker covers, so viewing distance matters more than print size.
