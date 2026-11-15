/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: ['gsap', '@gsap/react'],
  // a new CV's match takes 17-53 s and can wait behind another in the queue: keep the /api proxy open
  experimental: { proxyTimeout: 600000 },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
