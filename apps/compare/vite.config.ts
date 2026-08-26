import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  // Self-signed HTTPS so getUserMedia works from phones on the LAN;
  // NO_SSL=1 for a cert-warning-free local loop (localhost is secure).
  plugins: [process.env.NO_SSL ? undefined : basicSsl()].filter(Boolean),
  server: {
    host: true,
    fs: {
      allow: ["../.."],
    },
  },
  resolve: {
    // One three instance for everyone: mind-ar declares its own (older)
    // three, but content we add to its anchors must come from the same
    // module graph.
    dedupe: ["three"],
  },
  optimizeDeps: {
    exclude: ["tracear"],
  },
  worker: {
    format: "es",
  },
});
