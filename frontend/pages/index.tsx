import React, { useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import Header from '../components/Header';
import Hero from '../components/Hero';
import Features from '../components/Features';
import Footer from '../components/Footer';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

const RUBRIC_ROWS = [
  {
    axis: 'Skills',
    weight: 30,
    rule: 'Normalized skill and taxonomy overlap against job tags',
    fallback: 'Scored from CV skills',
  },
  {
    axis: 'Title',
    weight: 20,
    rule: 'Role family and seniority match with latest CV titles',
    fallback: 'Neutral 50 if omitted',
  },
  {
    axis: 'Experience',
    weight: 15,
    rule: 'Cumulative months of tenure up to the cutoff date',
    fallback: 'Neutral 50 if omitted',
  },
  {
    axis: 'Education',
    weight: 10,
    rule: 'Highest degree level matched against stated floor',
    fallback: 'Neutral 50 if omitted',
  },
  {
    axis: 'Industry',
    weight: 10,
    rule: 'Domain overlap between employers and job sector',
    fallback: 'Neutral 50 if omitted',
  },
  {
    axis: 'Location',
    weight: 8,
    rule: 'City or remote compatibility with office address',
    fallback: 'Neutral 50 if omitted',
  },
  {
    axis: 'Salary',
    weight: 7,
    rule: 'Range compatibility when both sides disclose pay',
    fallback: 'Neutral 75 if undisclosed',
  },
];

const HomePage: React.FC = () => {
  const rubricRef = useRef<HTMLElement | null>(null);
  const cachedSpeedRef = useRef<HTMLSpanElement | null>(null);
  const weightNumRefs = useRef<Array<HTMLSpanElement | null>>([]);

  useGSAP(
    () => {
      if (!rubricRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;

          const snapInt = gsap.utils.snap(1);

          const rubricTl = gsap.timeline({
            defaults: { ease: 'power3.out' },
            scrollTrigger: {
              trigger: '[data-rubric-grid]',
              start: 'top 86%',
              once: true,
            },
          });

          rubricTl
            .from('[data-rubric-card]', {
              y: 16,
              opacity: 0.85,
              immediateRender: false,
              duration: 0.5,
              stagger: 0.085,
              clearProps: 'transform,opacity',
            })
            .from(
              '[data-rubric-bar]',
              {
                scaleX: 0,
                immediateRender: false,
                transformOrigin: 'left center',
                duration: 0.62,
                stagger: 0.055,
                ease: 'power2.out',
                clearProps: 'transform',
              },
              '-=0.3'
            );

          if (cachedSpeedRef.current) {
            const speedObj = { val: 0 };
            rubricTl.to(
              speedObj,
              {
                val: 17,
                duration: 0.75,
                ease: 'power2.out',
                onUpdate: () => {
                  if (cachedSpeedRef.current) {
                    cachedSpeedRef.current.textContent =
                      speedObj.val.toFixed(0);
                  }
                },
              },
              0.18
            );
          }

          RUBRIC_ROWS.forEach((row, idx) => {
            const el = weightNumRefs.current[idx];
            if (!el) return;
            const state = { val: 0 };
            rubricTl.to(
              state,
              {
                val: row.weight,
                duration: 0.72,
                ease: 'power2.out',
                onUpdate: () => {
                  if (weightNumRefs.current[idx]) {
                    weightNumRefs.current[idx]!.textContent = String(
                      snapInt(state.val)
                    );
                  }
                },
              },
              0.2 + idx * 0.05
            );
          });

          return () => {
            if (cachedSpeedRef.current) {
              cachedSpeedRef.current.textContent = '17';
            }
            RUBRIC_ROWS.forEach((row, idx) => {
              if (weightNumRefs.current[idx]) {
                weightNumRefs.current[idx]!.textContent = String(row.weight);
              }
            });
          };
        },
        rubricRef
      );

      return () => mm.revert();
    },
    { scope: rubricRef }
  );

  return (
    <>
      <Head>
        <title>Matchly: CV-to-job matching</title>
        <meta
          name="description"
          content="Match a PDF, DOCX, or TXT resume against 361 job postings crawled from CareerViet and ITviec, or a pasted job description, with about 14 GB of GPU memory using fine-tuned Qwen2.5-7B extraction, seven weighted rules, and hybrid Qwen3-Embedding-4B ranking."
        />
      </Head>

      <div className="min-h-screen flex flex-col bg-ground text-body">
        <Header />

        <main id="main-content" className="flex-1">
          <Hero />

          <Features />

          {/* Section 3: Deterministic V3.1 Rubric & Runtime Contract (shares 5/7 split line with Hero) */}
          <section
            ref={rubricRef}
            id="rubric"
            aria-labelledby="rubric-heading"
            className="py-16 sm:py-24 border-t border-hairline"
          >
            <div className="wrap space-y-10 sm:space-y-12">
              <div className="max-w-3xl">
                <h2
                  id="rubric-heading"
                  data-rubric-heading
                  className="text-[28px] sm:text-[36px] leading-[1.14] text-ink"
                >
                  100-point scoring rubric
                </h2>
              </div>

              <div
                data-rubric-grid
                className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start"
              >
                {/* Left Column (span 5, aligned with Hero split): Runtime Spec + Visual + CTA */}
                <div className="lg:col-span-5 space-y-6 min-w-0">
                  <div
                    data-rubric-card
                    className="bento-card p-6 sm:p-8 space-y-5"
                  >
                    <h3 className="text-[20px] leading-snug text-ink">
                      Resume input & latency spec
                    </h3>

                    <dl className="space-y-3 text-[15px]">
                      <div className="pb-2.5 border-b border-hairline flex items-baseline justify-between gap-4">
                        <dt className="text-muted">Accepted files</dt>
                        <dd className="font-mono text-[14px] text-ink text-right">
                          .pdf, .docx, .txt ≤ 5&nbsp;MB
                        </dd>
                      </div>
                      <div className="pb-2.5 border-b border-hairline flex items-baseline justify-between gap-4">
                        <dt className="text-muted">Minimum text</dt>
                        <dd className="font-mono text-[14px] text-ink text-right">
                          200&nbsp;chars
                        </dd>
                      </div>
                      <div className="pb-2.5 border-b border-hairline flex items-baseline justify-between gap-4">
                        <dt className="text-muted">First CV run</dt>
                        <dd className="font-mono text-[14px] text-ink text-right">
                          20–60&nbsp;s
                        </dd>
                      </div>
                      <div className="flex items-baseline justify-between gap-4">
                        <dt className="text-muted">Cached CV</dt>
                        <dd className="font-mono text-[14px] text-good text-right tabular-nums">
                          ~<span ref={cachedSpeedRef}>17</span>&nbsp;s
                        </dd>
                      </div>
                    </dl>

                    <div
                      data-visual="cutoff-calibration"
                      className="bento-panel bento-panel--teal p-4"
                    >
                      <div className="ui-frag p-3.5 space-y-2 font-mono text-[13px]">
                        <div className="flex justify-between gap-2">
                          <span className="text-muted">V3.1 rule bands</span>
                          <span className="text-ink">43 · 58 / 100</span>
                        </div>
                        <div className="flex justify-between gap-2">
                          <span className="text-muted">hybrid_qwen_v1 fit</span>
                          <span className="text-ink">32.0 · 86.2 / 100</span>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-3 pt-1">
                      <Link
                        href="/analyze"
                        className="btn-primary w-full sm:w-auto"
                      >
                        <span>Upload a CV</span>
                        <ArrowRightIcon
                          className="w-4 h-4 m-arrow"
                          aria-hidden="true"
                        />
                      </Link>
                      <Link
                        href="/help"
                        className="btn-secondary w-full sm:w-auto"
                      >
                        <span>Open usage guide</span>
                      </Link>
                    </div>
                  </div>
                </div>

                {/* Right Column (span 7, aligned with Hero split): 7-Rule Specification Table + Animated Weight Bars */}
                <div
                  data-rubric-card
                  className="bento-card lg:col-span-7 p-6 sm:p-8 space-y-5 overflow-hidden min-w-0"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="text-[20px] leading-snug text-ink">
                      Seven V3.1 rule weights
                    </h3>
                    <span className="font-mono text-[13px] text-accent-text font-medium">
                      30 + 20 + 15 + 10 + 10 + 8 + 7 = 100
                    </span>
                  </div>

                  <div className="divide-y divide-hairline border-t border-b border-hairline">
                    {RUBRIC_ROWS.map((row, idx) => (
                      <div key={row.axis} className="py-3.5 space-y-2">
                        <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 sm:gap-4 items-baseline text-[15px]">
                          <div className="sm:col-span-3 flex items-center justify-between sm:justify-start gap-2">
                            <span className="font-medium text-ink">
                              {row.axis}
                            </span>
                            <span className="font-mono text-[12px] px-2 py-0.5 rounded bg-accent-soft border border-accent-line text-accent-text tabular-nums">
                              <span
                                ref={(el) => {
                                  weightNumRefs.current[idx] = el;
                                }}
                              >
                                {row.weight}
                              </span>
                              &nbsp;pt
                            </span>
                          </div>
                          <p className="sm:col-span-6 text-body leading-relaxed">
                            {row.rule}
                          </p>
                          <p className="sm:col-span-3 font-mono text-[12px] text-muted sm:text-right">
                            {row.fallback}
                          </p>
                        </div>
                        <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                          <div
                            data-rubric-bar
                            className="h-full rounded-full bar-brand origin-left"
                            style={{ width: `${(row.weight / 30) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="flex flex-wrap items-center gap-2.5 text-[13px] font-mono">
                    <span className="px-2.5 py-1 rounded bg-good-bg text-good border border-good-line">
                      Fit ≥ 86.2 Good Fit
                    </span>
                    <span className="px-2.5 py-1 rounded bg-pot-bg text-pot border border-pot-line">
                      32.0–86.1 Potential Fit
                    </span>
                    <span className="px-2.5 py-1 rounded bg-bad-bg text-bad border border-bad-line">
                      &lt; 32.0 No Fit
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>

        <Footer />
      </div>
    </>
  );
};

export default HomePage;
