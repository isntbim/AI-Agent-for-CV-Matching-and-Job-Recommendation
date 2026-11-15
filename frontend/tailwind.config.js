/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Manrope"', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['"Manrope"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'Consolas',
          'monospace',
        ],
      },
      colors: {
        ground: 'var(--bg)',
        surface: 'var(--surface)',
        card: 'var(--card)',
        frag: 'var(--frag)',
        ink: 'var(--ink)',
        body: 'var(--text)',
        muted: 'var(--muted)',
        hairline: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          ink: 'var(--accent-ink)',
          soft: 'var(--accent-soft)',
          text: 'var(--accent-text)',
          line: 'var(--accent-line)',
        },
        good: {
          DEFAULT: 'var(--good)',
          bg: 'var(--good-bg)',
          line: 'var(--good-line)',
        },
        pot: {
          DEFAULT: 'var(--pot)',
          bg: 'var(--pot-bg)',
          line: 'var(--pot-line)',
        },
        bad: {
          DEFAULT: 'var(--bad)',
          bg: 'var(--bad-bg)',
          line: 'var(--bad-line)',
        },
      },
      borderRadius: {
        bento: '20px',
        panel: '12px',
        ctrl: '8px',
      },
    },
  },
  plugins: [],
}
