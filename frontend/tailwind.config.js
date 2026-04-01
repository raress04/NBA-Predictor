/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0A0A0F',
        surface: {
          DEFAULT: '#111118',
          raised: '#1A1A24',
        },
        border: '#2A2A3A',
        accent: {
          primary: '#6C63FF',
          green: '#22C55E',
          amber: '#F59E0B',
          red: '#EF4444',
        },
        text: {
          primary: '#F1F1F5',
          secondary: '#8B8B9E',
          muted: '#4A4A5E',
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'glow': '0 0 40px rgba(108, 99, 255, 0.08)',
        'glow-hover': '0 0 60px rgba(108, 99, 255, 0.12)',
      }
    },
  },
  plugins: [],
}
