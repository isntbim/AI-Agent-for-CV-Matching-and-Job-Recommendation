import React, { useRef } from 'react';
import Link from 'next/link';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

const RULE_WEIGHTS = [
  { name: 'Skills', pts: 30 },
  { name: 'Title', pts: 20 },
  { name: 'Experience', pts: 15 },
  { name: 'Education', pts: 10 },
  { name: 'Industry', pts: 10 },
  { name: 'Location', pts: 8 },
  { name: 'Salary', pts: 7 },
];

const Features: React.FC = () => {
  const sectionRef = useRef<HTMLElement | null>(null);

  useGSAP(
    () => {
      if (!sectionRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;

          const cardsTl = gsap.timeline({
            defaults: { ease: 'power3.out' },
            scrollTrigger: {
              trigger: '[data-feat-grid]',
              start: 'top 86%',
              once: true,
            },
          });

          cardsTl
            .from('[data-feat-card]', {
              y: 16,
              opacity: 0.85,
              immediateRender: false,
              duration: 0.5,
              stagger: 0.075,
              clearProps: 'transform,opacity',
            })
            .from(
              '[data-feat-bar]',
              {
                scaleX: 0,
                immediateRender: false,
                transformOrigin: 'left center',
                duration: 0.62,
                stagger: 0.075,
                ease: 'power2.out',
                clearProps: 'transform',
              },
              '-=0.32'
            );
        },
        sectionRef
      );

      return () => mm.revert();
    },
    { scope: sectionRef }
  );

  return (
    <section
      ref={sectionRef}
      id="capabilities"
      aria-labelledby="capabilities-heading"
      className="py-16 sm:py-24 border-t border-hairline"
    >
      <div className="wrap space-y-10 sm:space-y-12">
        {/* Section Head */}
        <div className="max-w-3xl">
          <h2
            id="capabilities-heading"
            data-feat-heading
            className="text-[28px] sm:text-[36px] leading-[1.14] text-ink"
          >
            CareerViet matching pipeline in four stages.{' '}
            <span className="text-muted">
              Every score links back to literal text.
            </span>
          </h2>
        </div>

        {/* Asymmetric 12-Column SaaS Bento Grid */}
        <div
          data-feat-grid
          className="grid grid-cols-1 md:grid-cols-12 gap-6"
        >
          {/* Row 1, Wide Lead Highlight Card (span 8): Qwen2.5-7B + LangGraph Repair */}
          <article
            data-feat-card
            className="bento-card bento-card--highlight md:col-span-12 lg:col-span-8 p-6 sm:p-8 flex flex-col justify-between gap-7 min-w-0"
          >
            <div className="space-y-2 max-w-[54ch]">
              <h3 className="text-[20px] leading-snug text-ink">
                Qwen2.5-7B resume extractor
              </h3>
              <p className="text-[16px] leading-[1.55] text-body">
                QLoRA fine-tuned Qwen2.5-7B parses roles, skills, dates, and
                degrees, checked by a LangGraph validate-repair-retry state
                machine before scoring.
              </p>
            </div>

            <div
              data-visual="qwen-extractor"
              className="bento-panel bento-panel--cobalt p-4 sm:p-5"
            >
              <div className="ui-frag p-4 space-y-3 font-mono text-[13px]">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-hairline text-muted">
                  <span>qwen2.5-7b-v4-qlora · schema v3</span>
                  <span className="text-good">quote span verified</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-body">
                  <div className="space-y-1">
                    <div className="text-muted">work[0]</div>
                    <div className="text-ink">Web Developer (2020–present)</div>
                    <div className="text-muted">skills: 38 normalized tags</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-muted">validate_evidence()</div>
                    <div className="text-ink">
                      Paraphrased quotes rejected; exact CV substrings kept
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </article>

          {/* Row 1, Narrow Card (span 4): Hybrid Retrieval */}
          <article
            data-feat-card
            className="bento-card md:col-span-6 lg:col-span-4 p-6 sm:p-8 flex flex-col justify-between gap-7 min-w-0"
          >
            <div className="space-y-2">
              <h3 className="text-[20px] leading-snug text-ink">
                Top-20 CareerViet retrieval
              </h3>
              <p className="text-[16px] leading-[1.55] text-body">
                Multilingual BGE-M3 vectors and BM25 lexical scores fuse on CPU
                to narrow 361 postings to 20 candidates.
              </p>
            </div>

            <div
              data-visual="retrieval-funnel"
              className="bento-panel bento-panel--sky p-4"
            >
              <div className="ui-frag p-3.5 space-y-2.5 text-[13px] font-mono">
                <div className="flex items-center justify-between">
                  <span className="text-muted">CareerViet index</span>
                  <span className="text-ink font-medium">361 jobs</span>
                </div>
                <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                  <div
                    data-feat-bar
                    className="h-full w-full bg-ink/35 origin-left"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted">BGE-M3 + BM25 RRF</span>
                  <span className="text-accent-text font-medium">
                    Top 20 pool
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                  <div
                    data-feat-bar
                    className="h-full w-[18%] bar-brand rounded-full origin-left"
                  />
                </div>
              </div>
            </div>
          </article>

          {/* Row 2, Card 1 (span 4): 7 Rule Weights */}
          <Link
            href="/about#weights"
            data-feat-card
            className="bento-card bento-card-link md:col-span-6 lg:col-span-4 p-6 sm:p-8 flex flex-col justify-between gap-7 no-underline min-w-0"
          >
            <div className="space-y-2">
              <h3 className="text-[20px] leading-snug text-ink">
                Seven V3.1 rule weights
              </h3>
              <p className="text-[16px] leading-[1.55] text-body">
                Deterministic rules total 100 points and leave unsupported axes
                marked unscored instead of guessing.
              </p>
            </div>

            <div
              data-visual="rule-weights"
              className="bento-panel bento-panel--ice p-4"
            >
              <div className="ui-frag p-3.5 space-y-2 text-[13px]">
                <div className="flex flex-wrap gap-1.5 font-mono">
                  {RULE_WEIGHTS.map((w) => (
                    <span
                      key={w.name}
                      className="px-2 py-0.5 rounded-[6px] bg-surface border border-hairline text-ink"
                    >
                      {w.name}&nbsp;{w.pts}
                    </span>
                  ))}
                </div>
                <div className="text-muted font-mono pt-1">
                  Rule cutoffs: 43 · 58 / 100
                </div>
              </div>
            </div>
          </Link>

          {/* Row 2, Card 2 (span 4): Hybrid Ranking */}
          <Link
            href="/about#pipeline"
            data-feat-card
            className="bento-card bento-card-link md:col-span-6 lg:col-span-4 p-6 sm:p-8 flex flex-col justify-between gap-7 no-underline min-w-0"
          >
            <div className="space-y-2">
              <h3 className="text-[20px] leading-snug text-ink">
                Hybrid Qwen3-Embedding ranker
              </h3>
              <p className="text-[16px] leading-[1.55] text-body">
                Logistic weights blend rule scores with Qwen3-Embedding-4B
                whole-text, job-title, and TF-IDF cosines.
              </p>
            </div>

            <div
              data-visual="hybrid-ranker"
              className="bento-panel bento-panel--teal p-4"
            >
              <div className="ui-frag p-3.5 space-y-2 text-[13px] font-mono">
                <div className="flex justify-between">
                  <span className="text-muted">text_cos</span>
                  <span className="text-ink">0.71</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">title_cos</span>
                  <span className="text-ink">0.81</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted">tfidf_cos</span>
                  <span className="text-ink">0.24</span>
                </div>
              </div>
            </div>
          </Link>

          {/* Row 2, Card 3 (span 4): Local 12 GB GPU */}
          <Link
            href="/about#runtime"
            data-feat-card
            className="bento-card bento-card-link md:col-span-6 lg:col-span-4 p-6 sm:p-8 flex flex-col justify-between gap-7 no-underline min-w-0"
          >
            <div className="space-y-2">
              <h3 className="text-[20px] leading-snug text-ink">
                Single 12&nbsp;GB GPU runtime
              </h3>
              <p className="text-[16px] leading-[1.55] text-body">
                Ollama keeps extractor and embedding models co-resident in
                9.7&nbsp;GB VRAM, with SHA-256 extraction caching.
              </p>
            </div>

            <div
              data-visual="vram-budget"
              className="bento-panel bento-panel--cobalt p-4"
            >
              <div className="ui-frag p-3.5 space-y-2.5 text-[13px] font-mono">
                <div className="flex justify-between">
                  <span className="text-muted">VRAM in use</span>
                  <span className="text-ink">9.7 / 12.0&nbsp;GB</span>
                </div>
                <div className="h-2 rounded-full bg-surface overflow-hidden flex">
                  <div
                    data-feat-bar
                    className="h-full bar-brand origin-left"
                    style={{ width: '55%' }}
                    title="qwen-v4: 6.6 GB"
                  />
                  <div
                    data-feat-bar
                    className="h-full bg-ink/40 origin-left"
                    style={{ width: '26%' }}
                    title="qwen3-embedding:4b: 3.1 GB"
                  />
                </div>
                <div className="text-muted">
                  6.6&nbsp;GB extractor · 3.1&nbsp;GB emb
                </div>
              </div>
            </div>
          </Link>
        </div>
      </div>
    </section>
  );
};

export default Features;
