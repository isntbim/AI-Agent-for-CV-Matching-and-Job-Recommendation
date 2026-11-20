import React, { useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

const Custom404: React.FC = () => {
  const mainRef = useRef<HTMLElement | null>(null);

  useGSAP(
    () => {
      if (!mainRef.current) return;
      const mm = gsap.matchMedia();
      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          gsap.from('[data-404-card]', {
            y: 14,
            opacity: 0.85,
            duration: 0.45,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
          });
        },
        mainRef
      );
      return () => mm.revert();
    },
    { scope: mainRef }
  );

  return (
    <>
      <Head>
        <title>404 — Page not found | Matchly</title>
        <meta
          name="description"
          content="The requested page could not be found in Matchly."
        />
      </Head>

      <div className="min-h-screen flex flex-col bg-ground text-body">
        <Header />

        <main
          ref={mainRef}
          id="main-content"
          className="flex-1 pt-[84px] sm:pt-[108px] pb-16 flex items-center"
        >
          <div className="wrap">
            <div
              data-404-card
              className="bento-card bento-card--highlight p-8 sm:p-12 max-w-2xl space-y-6"
            >
              <p className="inline-flex items-center px-2.5 py-1 rounded-full bg-accent-soft border border-accent-line font-mono text-[13px] text-accent-text">
                Error 404
              </p>
              <h1 className="text-[32px] sm:text-[42px] leading-[1.1] text-ink">
                <span className="text-gradient">Page not found</span>
              </h1>
              <p className="text-[16px] text-body leading-relaxed">
                This URL does not match any page in Matchly. Jump directly to
                the CV matcher or return to the overview.
              </p>

              <div className="flex flex-col sm:flex-row gap-3 pt-1">
                <Link href="/analyze" className="btn-primary w-full sm:w-auto">
                  <span>Upload a CV</span>
                  <ArrowRightIcon
                    className="w-4 h-4 m-arrow"
                    aria-hidden="true"
                  />
                </Link>
                <Link href="/" className="btn-secondary w-full sm:w-auto">
                  <span>Back to overview</span>
                </Link>
                <Link href="/help" className="btn-secondary w-full sm:w-auto">
                  <span>Open usage guide</span>
                </Link>
              </div>
            </div>
          </div>
        </main>

        <Footer hideClosingCta />
      </div>
    </>
  );
};

export default Custom404;
