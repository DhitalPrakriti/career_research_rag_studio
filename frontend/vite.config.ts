import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev the UI runs on 5173 and proxies /api to the FastAPI server on 8000, so the
// browser makes same-origin requests. In production FastAPI serves this build itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
