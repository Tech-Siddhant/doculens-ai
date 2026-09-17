import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#f7f9fb",
        surface: {
          DEFAULT: "#f7f9fb",
          elevated: "#ffffff",
          low: "#f2f4f6",
          container: "#eceef0",
          high: "#e6e8ea",
        },
        brand: {
          DEFAULT: "#0051d5",
          hover: "#003ea8",
          light: "#eff6ff",
          container: "#316bf3",
        },
        text: {
          primary: "#0f172a",
          secondary: "#475569",
          tertiary: "#94a3b8",
        },
        border: {
          subtle: "#e2e8f0",
          strong: "#0f172a",
          muted: "#cbd5e1",
        },
        status: {
          ready: "#10b981",
          processing: "#f59e0b",
          failed: "#ef4444",
          attention: "#f59e0b",
          neutral: "#94a3b8",
        },
      },
      fontFamily: {
        sans: [
          "Hanken Grotesk",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        heading: [
          "Space Grotesk",
          "-apple-system",
          "BlinkMacSystemFont",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
