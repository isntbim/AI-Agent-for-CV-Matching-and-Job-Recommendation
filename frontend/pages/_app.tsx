import '@/styles/globals.css'
import { useEffect } from 'react'
import type { AppProps } from 'next/app'
import Head from 'next/head'
import { useRouter } from 'next/router'
import { Toaster } from 'react-hot-toast'
import ErrorBoundary from '@/components/ErrorBoundary'
import { ensureGsapRegistered, ScrollTrigger } from '@/utils/gsap'

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter()

  useEffect(() => {
    ensureGsapRegistered()
    const handleRouteComplete = () => {
      requestAnimationFrame(() => {
        ScrollTrigger.refresh()
      })
    }
    router.events.on('routeChangeComplete', handleRouteComplete)
    return () => {
      router.events.off('routeChangeComplete', handleRouteComplete)
    }
  }, [router.events])

  return (
    <>
      <Head>
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, viewport-fit=cover"
        />
        <meta property="og:type" content="website" />
        <meta
          property="og:title"
          content="Matchly: CV-to-job matching"
        />
        <meta
          property="og:description"
          content="Rank a PDF, DOCX, or TXT resume against 361 crawled job postings or a pasted job description, using Qwen2.5-7B extraction, seven rule weights, and literal quotes (about 14 GB of GPU memory in total)."
        />
        <meta property="og:image" content="/icon-512.png" />
        <meta name="twitter:card" content="summary" />
      </Head>
      <ErrorBoundary>
        <Component {...pageProps} />
        <Toaster
          position="bottom-right"
          containerStyle={{ inset: 16 }}
          toastOptions={{
            duration: 3500,
            style: {
              background: 'var(--card)',
              color: 'var(--ink)',
              border: '1px solid var(--line-strong)',
              borderRadius: '10px',
              fontSize: '14px',
              maxWidth: 'calc(100vw - 32px)',
              boxShadow: '0 8px 24px rgba(11, 26, 51, 0.14)',
            },
            success: {
              iconTheme: {
                primary: '#1d4ed8',
                secondary: '#ffffff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#b42318',
                secondary: '#ffffff',
              },
            },
          }}
        />
      </ErrorBoundary>
    </>
  )
}
