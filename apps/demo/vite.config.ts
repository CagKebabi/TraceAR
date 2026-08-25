import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

export default defineConfig({
  // Self-signed HTTPS so getUserMedia works from phones on the LAN.
  // localhost is a secure context even over plain HTTP, so NO_SSL=1 gives a
  // cert-warning-free local dev loop.
  plugins: process.env.NO_SSL ? [] : [basicSsl()],
  server: {
    host: true,
    fs: {
      // Monorepo: allow serving the linked SDK source + wasm assets.
      allow: ["../.."],
    },
  },
  optimizeDeps: {
    // Consume the SDK as source (it ships .ts + a wasm asset — prebundling breaks both).
    exclude: ["tracear"],
  },
  worker: {
    format: "es",
  },
});
