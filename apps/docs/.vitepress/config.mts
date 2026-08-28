import { defineConfig } from "vitepress";

export default defineConfig({
  title: "TraceAR",
  description:
    "High-performance, jitter-free image tracking for the mobile web — Rust/WASM core, ~95 KB, free and MIT-licensed.",
  base: process.env.DOCS_BASE ?? "/TraceAR/docs/",
  appearance: "force-dark",
  lastUpdated: true,
  themeConfig: {
    nav: [
      { text: "Guide", link: "/guide/getting-started", activeMatch: "/guide/" },
      { text: "API", link: "/reference/tracear", activeMatch: "/reference/" },
      { text: "Live demo", link: "https://cagkebabi.github.io/TraceAR/demo/" },
      { text: "npm", link: "https://www.npmjs.com/package/@tracear/sdk" },
    ],
    sidebar: {
      "/guide/": [
        {
          text: "Guide",
          items: [
            { text: "Getting started", link: "/guide/getting-started" },
            { text: "Markers", link: "/guide/markers" },
            { text: "Rendering with three.js", link: "/guide/three" },
            { text: "Tracking & poses", link: "/guide/tracking-and-poses" },
            { text: "Performance & browsers", link: "/guide/performance" },
          ],
        },
        {
          text: "API reference",
          items: [{ text: "Overview", link: "/reference/tracear" }],
        },
      ],
      "/reference/": [
        {
          text: "API reference",
          items: [
            { text: "Tracear", link: "/reference/tracear" },
            { text: "TracearThree", link: "/reference/three" },
            { text: "Marker compiler & CLI", link: "/reference/compiler" },
          ],
        },
        {
          text: "Guide",
          items: [{ text: "Getting started", link: "/guide/getting-started" }],
        },
      ],
    },
    socialLinks: [{ icon: "github", link: "https://github.com/CagKebabi/TraceAR" }],
    search: { provider: "local" },
    editLink: {
      pattern: "https://github.com/CagKebabi/TraceAR/edit/main/apps/docs/:path",
      text: "Edit this page on GitHub",
    },
    footer: {
      message: "Released under the MIT License.",
      copyright: "© 2026 Ahmet Güleş",
    },
  },
});
