/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark Theme Colors
        dark: {
          bg: {
            primary: '#0a0a0a',
            secondary: '#111111',
            tertiary: '#1a1a1a',
            card: '#161616',
            hover: '#202020',
          },
          text: {
            primary: '#ffffff',
            secondary: '#a3a3a3',
            muted: '#737373',
          },
          border: {
            primary: '#262626',
            secondary: '#404040',
            accent: '#3b82f6',
          },
          accent: {
            primary: '#3b82f6',
            secondary: '#10b981',
            tertiary: '#8b5cf6',
            warning: '#f59e0b',
            error: '#ef4444',
          }
        },
        // Keep existing colors for compatibility
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'dark-glow': 'dark-glow 3s ease-in-out infinite',
        'dark-pulse': 'dark-pulse 2s ease-in-out infinite',
        'dark-slide-up': 'dark-slide-up 0.6s ease-out',
        'dark-slide-in': 'dark-slide-in 0.6s ease-out',
        'dark-float': 'dark-float 4s ease-in-out infinite',
        'dark-shimmer': 'dark-shimmer 2s ease-in-out infinite',
      },
      keyframes: {
        'dark-glow': {
          '0%, 100%': { 
            boxShadow: '0 0 20px rgba(59, 130, 246, 0.2)' 
          },
          '50%': { 
            boxShadow: '0 0 30px rgba(59, 130, 246, 0.4)' 
          },
        },
        'dark-pulse': {
          '0%, 100%': { 
            opacity: '1',
            transform: 'scale(1)' 
          },
          '50%': { 
            opacity: '0.8',
            transform: 'scale(1.02)' 
          },
        },
        'dark-slide-up': {
          'from': {
            opacity: '0',
            transform: 'translateY(20px)'
          },
          'to': {
            opacity: '1',
            transform: 'translateY(0)'
          },
        },
        'dark-slide-in': {
          'from': {
            opacity: '0',
            transform: 'translateX(-20px)'
          },
          'to': {
            opacity: '1',
            transform: 'translateX(0)'
          },
        },
        'dark-float': {
          '0%, 100%': {
            transform: 'translateY(0px)'
          },
          '50%': {
            transform: 'translateY(-10px)'
          },
        },
        'dark-shimmer': {
          '0%': {
            backgroundPosition: '-200% center'
          },
          '100%': {
            backgroundPosition: '200% center'
          },
        },
      },
      backgroundImage: {
        'dark-gradient-primary': 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
        'dark-gradient-accent': 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
        'dark-gradient-card': 'linear-gradient(145deg, #161616 0%, #1a1a1a 100%)',
      },
      boxShadow: {
        'dark-sm': '0 1px 2px 0 rgba(0, 0, 0, 0.5)',
        'dark-md': '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
        'dark-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2)',
        'dark-xl': '0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2)',
        'dark-glow': '0 0 20px rgba(59, 130, 246, 0.3)',
      },
      backdropBlur: {
        'dark': '20px',
      },
      borderRadius: {
        'dark': '16px',
      }
    },
  },
  plugins: [],
}
