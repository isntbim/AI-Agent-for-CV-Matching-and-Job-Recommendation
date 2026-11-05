import { Html, Head, Main, NextScript } from 'next/document'

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
        <meta name="description" content="AI-powered resume and job description matching system" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <body className="font-inter antialiased">
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
