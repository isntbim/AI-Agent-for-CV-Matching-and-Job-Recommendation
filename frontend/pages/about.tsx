import React, { useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

const PIPELINE_STAGES = [
  {
    step: '01',
    title: 'Structured extraction & quote verification',
    model: 'qwen-v4 (QLoRA Qwen2.5-7B) + LangGraph',
    span: 'lg:col-span-6',
    summary:
      'Parses raw CV text (.pdf, .docx, .txt) into schema v3 JSON covering work roles, date spans, normalized skills, degrees, location, and salary expectations.',
    detail:
      'A LangGraph validate-repair-retry loop checks JSON validity and verifies that every extracted quote is an exact substring of the source CV.',
  },
  {
    step: '02',
    title: 'Hybrid candidate retrieval',
    model: 'BAAI/bge-m3 + BM25 (CPU)',
    span: 'lg:col-span-6',
    summary:
      'Searches the 361 indexed job postings (crawled from CareerViet and ITviec) by combining multilingual dense cosine similarity with BM25 lexical term matching.',
    detail:
      'Reciprocal rank fusion narrows 361 postings down to a top-20 candidate pool in milliseconds on CPU without touching GPU VRAM.',
  },
  {
    step: '03',
    title: 'Deterministic seven-axis V3.1 scoring',
    model: 'Rule scorer (100 points total)',
    span: 'lg:col-span-6',
    summary:
      'Evaluates each of the 20 retrieved jobs on Skills (30), Title (20), Experience (15), Education (10), Industry (10), Location (8), and Salary (7).',
    detail:
      'Dimensions lacking explicit evidence in either the CV or the job posting receive a neutral value and are flagged in the explanation rather than guessed.',
  },
  {
    step: '04',
    title: 'Hybrid logistic reranking',
    model: 'hybrid_qwen_v1 · Qwen3-Embedding-4B + TF-IDF',
    span: 'lg:col-span-6',
    summary:
      'Combines the seven rule scores with Qwen3-Embedding-4B whole-text cosine, job-title cosine, and corpus TF-IDF cosine via calibrated logistic weights.',
    detail:
      'Produces the final 0–100 fit score, assigns Good Fit / Potential Fit / No Fit grades, and compiles plain-English explanations with literal quotes.',
  },
];

const WEIGHT_SPEC = [
  { name: 'Skills', pts: 30, note: 'Required & bonus skill tag overlap' },
  { name: 'Title', pts: 20, note: 'Role family & seniority level match' },
  { name: 'Experience', pts: 15, note: 'Total work tenure up to cutoff date' },
  { name: 'Education', pts: 10, note: 'Degree floor & field alignment' },
  { name: 'Industry', pts: 10, note: 'Employer domain & job sector match' },
  { name: 'Location', pts: 8, note: 'City or remote work compatibility' },
  { name: 'Salary', pts: 7, note: 'Disclosed range overlap (neutral 75)' },
];

const API_ROUTES = [
  {
    method: 'GET',
    path: '/api/health',
    desc: 'Ollama extractor status, corpus size (361), and active ranker',
  },
  {
    method: 'POST',
    path: '/api/match/stream',
    desc: 'Multipart SSE stream emitting token, step, and result events',
  },
  {
    method: 'POST',
    path: '/api/match',
    desc: 'Synchronous multipart CV match endpoint',
  },
  {
    method: 'GET',
    path: '/api/history',
    desc: 'Recent matching runs stored in local SQLite',
  },
  {
    method: 'GET',
    path: '/api/history/{run_id}',
    desc: 'Full saved response and feedback for a specific run',
  },
  {
    method: 'POST',
    path: '/api/feedback',
    desc: 'Persist thumbs vote (+1 / -1) and optional grade override',
  },
];

const AboutPage: React.FC = () => {
  const mainRef = useRef<HTMLElement | null>(null);
  const weightPtRefs = useRef<Array<HTMLSpanElement | null>>([]);

  useGSAP(
    () => {
      if (!mainRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          const snapInt = gsap.utils.snap(1);

          gsap.from('[data-about-hero]', {
            y: 14,
            opacity: 0,
            duration: 0.45,
            stagger: 0.055,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
          });

          gsap.from('[data-pipeline-card]', {
            y: 16,
            opacity: 0.85,
            immediateRender: false,
            duration: 0.48,
            stagger: 0.075,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
            scrollTrigger: {
              trigger: '#pipeline',
              start: 'top 86%',
              once: true,
            },
          });

          const weightsTl = gsap.timeline({
            defaults: { ease: 'power3.out' },
            scrollTrigger: {
              trigger: '#weights',
              start: 'top 86%',
              once: true,
            },
          });

          weightsTl
            .from('[data-weights-card]', {
              y: 16,
              opacity: 0.85,
              immediateRender: false,
              duration: 0.48,
              stagger: 0.085,
              clearProps: 'transform,opacity',
            })
            .from(
              '[data-about-bar]',
              {
                scaleX: 0,
                immediateRender: false,
                transformOrigin: 'left center',
                duration: 0.58,
                stagger: 0.05,
                ease: 'power2.out',
                clearProps: 'transform',
              },
              '-=0.28'
            );

          WEIGHT_SPEC.forEach((item, idx) => {
            const el = weightPtRefs.current[idx];
            if (!el) return;
            const state = { val: 0 };
            weightsTl.to(
              state,
              {
                val: item.pts,
                duration: 0.68,
                ease: 'power2.out',
                onUpdate: () => {
                  if (weightPtRefs.current[idx]) {
                    weightPtRefs.current[idx]!.textContent = String(
                      snapInt(state.val)
                    );
                  }
                },
              },
              0.16 + idx * 0.05
            );
          });

          gsap.from('[data-runtime-card]', {
            y: 16,
            opacity: 0.85,
            immediateRender: false,
            duration: 0.48,
            stagger: 0.085,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
            scrollTrigger: {
              trigger: '#runtime',
              start: 'top 86%',
              once: true,
            },
          });

          return () => {
            WEIGHT_SPEC.forEach((item, idx) => {
              if (weightPtRefs.current[idx]) {
                weightPtRefs.current[idx]!.textContent = String(item.pts);
              }
            });
          };
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
        <title>Architecture — Matchly</title>
        <meta
          name="description"
          content="Technical architecture of the CV-JD matching pipeline: QLoRA Qwen2.5-7B extraction, BGE-M3 + BM25 retrieval, seven V3.1 rule weights, and hybrid_qwen_v1 ranking, with about 14 GB of GPU memory."
        />
      </Head>

      <div className="min-h-screen flex flex-col bg-ground text-body">
        <Header />

        <main
          ref={mainRef}
          id="main-content"
          className="flex-1 pt-[84px] sm:pt-[108px] pb-16"
        >
          {/* Intro Header */}
          <section className="pb-14 sm:pb-20">
            <div className="wrap space-y-5">
              <p
                data-about-hero
                className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-soft border border-accent-line font-mono text-[13px] text-accent-text"
              >
                Capstone FA26AI45 · Local inference specification
              </p>
              <h1
                data-about-hero
                className="text-[34px] sm:text-[46px] lg:text-[52px] leading-[1.08] text-ink max-w-4xl"
              >
                Architecture of the{' '}
                <span className="text-gradient">
                  CV-to-job matching pipeline
                </span>
              </h1>
              <p
                data-about-hero
                className="text-[17px] sm:text-[18px] leading-[1.55] text-body max-w-[60ch]"
              >
                Matchly pairs language-model extraction with deterministic
                seven-axis rule scoring and embedding cosines so every ranked
                job can be audited back to verbatim text. The three models use
                about 14 GB of GPU memory.
              </p>
              <div
                data-about-hero
                className="pt-1 flex flex-col sm:flex-row gap-3"
              >
                <Link href="/analyze" className="btn-primary w-full sm:w-auto">
                  <span>Upload a CV</span>
                  <ArrowRightIcon
                    className="w-4 h-4 m-arrow"
                    aria-hidden="true"
                  />
                </Link>
                <Link href="/help" className="btn-secondary w-full sm:w-auto">
                  <span>Open usage guide</span>
                </Link>
              </div>
            </div>
          </section>

          {/* Section 1: Four-Stage Pipeline Bento */}
          <section
            id="pipeline"
            aria-labelledby="pipeline-heading"
            className="py-14 sm:py-20 border-t border-hairline"
          >
            <div className="wrap space-y-10">
              <div className="max-w-3xl space-y-2">
                <h2
                  id="pipeline-heading"
                  className="text-[26px] sm:text-[36px] leading-[1.14] text-ink"
                >
                  Four sequential stages per run
                </h2>
                <p className="text-[16px] text-body">
                  All 361 job postings are pre-extracted offline; a
                  live request only extracts the uploaded CV once and caches it
                  by SHA-256 hash.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {PIPELINE_STAGES.map((stage) => (
                  <article
                    key={stage.step}
                    data-pipeline-card
                    className={`bento-card ${stage.span} p-6 sm:p-8 flex flex-col justify-between gap-5 min-w-0`}
                  >
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[12px]">
                        <span className="px-2 py-0.5 rounded bg-accent-soft border border-accent-line text-accent-text">
                          Stage {stage.step}
                        </span>
                        <span className="text-muted">{stage.model}</span>
                      </div>
                      <h3 className="text-[20px] leading-snug text-ink">
                        {stage.title}
                      </h3>
                      <p className="text-[15px] text-body leading-relaxed">
                        {stage.summary}
                      </p>
                    </div>

                    <p className="pt-4 border-t border-hairline text-[14px] text-muted leading-relaxed">
                      {stage.detail}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          </section>

          {/* Section 2: Seven Rule Weights & Cutoffs */}
          <section
            id="weights"
            aria-labelledby="weights-heading"
            className="py-14 sm:py-20 border-t border-hairline"
          >
            <div className="wrap space-y-10">
              <div className="max-w-3xl space-y-2">
                <h2
                  id="weights-heading"
                  className="text-[26px] sm:text-[36px] leading-[1.14] text-ink"
                >
                  Seven rule weights and grade cutoffs
                </h2>
                <p className="text-[16px] text-body">
                  Weights sum to 100 points and reflect how strongly each signal
                  separates qualified candidates across job postings.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                <div
                  data-weights-card
                  className="bento-card bento-card--highlight lg:col-span-7 p-6 sm:p-8 space-y-4 min-w-0"
                >
                  {WEIGHT_SPEC.map((item, idx) => (
                    <div key={item.name} className="space-y-1.5">
                      <div className="flex items-baseline justify-between gap-3 text-[14px]">
                        <div>
                          <span className="font-medium text-ink">
                            {item.name}
                          </span>
                          <span className="text-muted ml-2 text-[13px]">
                            · {item.note}
                          </span>
                        </div>
                        <span className="font-mono text-[13px] font-medium text-ink tabular-nums shrink-0">
                          <span
                            ref={(el) => {
                              weightPtRefs.current[idx] = el;
                            }}
                          >
                            {item.pts}
                          </span>{' '}
                          / 100
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-surface overflow-hidden">
                        <div
                          data-about-bar
                          className="h-full rounded-full bar-brand origin-left"
                          style={{ width: `${(item.pts / 30) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <div
                  data-weights-card
                  className="bento-card lg:col-span-5 p-6 sm:p-8 space-y-5 min-w-0"
                >
                  <h3 className="text-[20px] leading-snug text-ink">
                    Unscored axes & decision bands
                  </h3>
                  <p className="text-[15px] text-body leading-relaxed">
                    Crawled job postings often omit salary ranges, explicit
                    degree requirements, or years of tenure. Instead of
                    penalizing the candidate or hallucinating missing fields,
                    the scorer assigns a neutral midpoint (50, or 75 for salary)
                    and lists every unscored axis in the explanation.
                  </p>
                  <dl className="space-y-2.5 pt-2 border-t border-hairline font-mono text-[13px]">
                    <div className="flex items-center justify-between py-1">
                      <dt className="text-good">Good Fit</dt>
                      <dd className="text-ink tabular-nums">
                        fit ≥ 86.2 (rule ≥ 58)
                      </dd>
                    </div>
                    <div className="flex items-center justify-between py-1">
                      <dt className="text-pot">Potential Fit</dt>
                      <dd className="text-ink tabular-nums">
                        fit 32.0–86.1 (rule 43–57.9)
                      </dd>
                    </div>
                    <div className="flex items-center justify-between py-1">
                      <dt className="text-bad">No Fit</dt>
                      <dd className="text-ink tabular-nums">
                        fit &lt; 32.0 (rule &lt; 43)
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>
          </section>

          {/* Section 3: GPU memory budget & FastAPI Contract */}
          <section
            id="runtime"
            aria-labelledby="runtime-heading"
            className="py-14 sm:py-20 border-t border-hairline"
          >
            <div className="wrap space-y-10">
              <div className="max-w-3xl space-y-2">
                <h2
                  id="runtime-heading"
                  className="text-[26px] sm:text-[36px] leading-[1.14] text-ink"
                >
                  GPU memory budget and FastAPI routes
                </h2>
                <p className="text-[16px] text-body">
                  The models run in Ollama on a Runpod serverless GPU (or a local
                  GPU) behind FastAPI; no commercial LLM API is called.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                <div
                  data-runtime-card
                  className="bento-card lg:col-span-5 p-6 sm:p-8 space-y-4 min-w-0"
                >
                  <h3 className="text-[20px] leading-snug text-ink">
                    About 14 GB VRAM, three models
                  </h3>
                  <dl className="space-y-3 text-[14px]">
                    <div className="pb-2.5 border-b border-hairline flex justify-between gap-3">
                      <dt className="text-muted">Extractor (qwen-v4)</dt>
                      <dd className="font-mono text-ink">6.2 GB VRAM</dd>
                    </div>
                    <div className="pb-2.5 border-b border-hairline flex justify-between gap-3">
                      <dt className="text-muted">Explainer (qwen-o5)</dt>
                      <dd className="font-mono text-ink">4.8 GB VRAM</dd>
                    </div>
                    <div className="pb-2.5 border-b border-hairline flex justify-between gap-3">
                      <dt className="text-muted">
                        Ranker (qwen3-embedding:4b)
                      </dt>
                      <dd className="font-mono text-ink">3.2 GB VRAM</dd>
                    </div>
                    <div className="pb-2.5 border-b border-hairline flex justify-between gap-3">
                      <dt className="text-muted">Retrieval (BGE-M3 + BM25)</dt>
                      <dd className="font-mono text-ink">CPU RAM</dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt className="text-muted">Concurrency guard</dt>
                      <dd className="font-mono text-ink">Async GPU lock</dd>
                    </div>
                  </dl>
                </div>

                <div
                  data-runtime-card
                  className="bento-card lg:col-span-7 p-6 sm:p-8 space-y-4 min-w-0"
                >
                  <h3 className="text-[20px] leading-snug text-ink">
                    Backend HTTP endpoints
                  </h3>
                  <div className="divide-y divide-hairline border-t border-b border-hairline">
                    {API_ROUTES.map((route) => (
                      <div
                        key={route.path}
                        className="py-3 flex flex-col sm:flex-row sm:items-baseline justify-between gap-1.5 sm:gap-4 text-[14px]"
                      >
                        <div className="flex items-center gap-2 font-mono text-[13px] shrink-0">
                          <span className="px-1.5 py-0.5 rounded bg-accent-soft border border-accent-line text-accent-text">
                            {route.method}
                          </span>
                          <span className="text-ink">{route.path}</span>
                        </div>
                        <span className="text-muted sm:text-right">
                          {route.desc}
                        </span>
                      </div>
                    ))}
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

export default AboutPage;
