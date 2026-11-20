import { Html, Head, Main, NextScript } from 'next/document'

const THEME_INIT_SCRIPT = `(function(){try{var s=localStorage.getItem('resumatch-theme');var d=s?s==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;if(d){document.documentElement.classList.add('dark');var m=document.querySelector('meta[name="theme-color"]');if(m)m.setAttribute('content','#070d1a');}}catch(e){}})();`

export default function Document() {
  return (
    <Html
      lang="en"
      data-type="software"
      data-radius="soft"
      data-buttons="solid"
    >
      <Head>
        <meta name="page-type" content="software" />
        <meta name="site-kind" content="software" />
        <meta name="theme-color" content="#f4f7fc" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
        <link rel="icon" href="/favicon.ico" sizes="48x48" />
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/site.webmanifest" />
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </Head>
      <body className="font-sans antialiased">
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
