import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@wllama/wllama/esm/wasm-from-cdn.js': fileURLToPath(
        new URL('./src/wasm-from-cdn.js', import.meta.url)
      ),
    },
  },
});
