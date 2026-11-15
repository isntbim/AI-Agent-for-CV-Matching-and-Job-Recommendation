import React, { useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

const QUICK_SPECS = [
  {
    label: 'Accepted formats',
    value: '.pdf, .docx, .txt',
    detail:
      'Up to 5 MB with at least 200 selectable characters of resume text.',
    highlight: false,
  },
  {
    label: 'First-run latency',
    value: '20–60 s',
    detail:
      'Streams live Qwen2.5-7B tokens over SSE while extracting CV structure.',
    highlight: false,
  },
  {
    label: 'Cached latency',
    value: '~0.9 s',
    detail:
      'Identical CV text + cutoff date hits the local SHA-256 extraction cache.',
    highlight: false,
  },
  {
    label: 'Corpus & ranker',
    value: '361 jobs',
    detail: 'Top 20 retrieved via BGE-M3 + BM25, reranked by hybrid_qwen_v1.',
    highlight: true,
  },
];

const FAQS = [
  {
    question: 'What file formats and sizes are accepted?',
    answer:
      'Upload text-based .pdf, Microsoft Word .docx, or plain-text .txt files up to 5 MB, or switch to Paste CV text and enter at least 200 characters directly. Scanned image-only PDFs contain no selectable text and are rejected with an explicit message.',
  },
  {
    question: 'Why does a brand-new CV take 20 to 60 seconds?',
    answer:
      'Every new CV is parsed locally on a single 12 GB GPU by the QLoRA fine-tuned Qwen2.5-7B extractor and verified by a LangGraph repair loop. Once extracted, its structured JSON is cached on disk by SHA-256 hash and cutoff date, so repeat runs return in about one second.',
  },
  {
    question: 'How is the 0–100 fit score calculated?',
    answer:
      'First, deterministic V3.1 rules score seven weighted dimensions: Skills (30), Title (20), Experience (15), Education (10), Industry (10), Location (8), and Salary (7). Then the hybrid_qwen_v1 ranker combines those rule scores with Qwen3-Embedding-4B whole-text cosine, job-title cosine, and corpus TF-IDF cosine.',
  },
  {
    question: 'What does “not scored” mean on the breakdown or radar?',
    answer:
      'When a job posting or CV omits evidence for an axis (such as undisclosed salary or unspecified degree requirements), the scorer assigns a neutral value (50, or 75 for salary) and marks that dimension as not scored rather than guessing.',
  },
  {
    question: 'What does “Experience up to” (as_of date) control?',
    answer:
      'Roles marked “Present” on a resume need a reference date to calculate total months of tenure. Changing Experience up to recomputes cumulative work experience as of that calendar date.',
  },
  {
    question:
      'What do “Matching unavailable · saved runs ok” or “API offline” in the header mean?',
    answer:
      '“361 jobs · Qwen up” means both FastAPI and Ollama are ready for live CV matching. “Matching unavailable · saved runs ok” means FastAPI is reachable on port 8000 while Ollama is stopped: new and cached CVs cannot rank because Ranker.scores() requests fresh Qwen3-Embedding-4B vectors on every match, while previously saved runs in Recent runs remain available from SQLite. “API offline” means the FastAPI server at 127.0.0.1:8000 is not running.',
  },
];

const HelpPage: React.FC = () => {
  const mainRef = useRef<HTMLElement | null>(null);

  useGSAP(
    () => {
      if (!mainRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate || !mainRef.current) return;

          gsap.from('[data-help-hero]', {
            y: 14,
            opacity: 0,
            duration: 0.45,
            stagger: 0.055,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
          });

          gsap.from('[data-spec-card]', {
            y: 14,
            opacity: 0.85,
            immediateRender: false,
            duration: 0.46,
            stagger: 0.065,
            ease: 'power3.out',
            clearProps: 'transform,opacity',
            scrollTrigger: {
              trigger: '[data-spec-grid]',
              start: 'top 88%',
              once: true,
            },
          });

          const faqCards = gsap.utils.toArray<HTMLElement>(
            '[data-faq-card]',
            mainRef.current
          );
          faqCards.forEach((card, idx) => {
            gsap.from(card, {
              y: 14,
              opacity: 0.85,
              immediateRender: false,
              duration: 0.44,
              delay: (idx % 2) * 0.06,
              ease: 'power3.out',
              clearProps: 'transform,opacity',
              scrollTrigger: {
                trigger: card,
                start: 'top 90%',
                once: true,
              },
            });
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
        <title>Usage guide — ResuMatch</title>
        <meta
          name="description"
          content="Usage guide and common questions for running ResuMatch locally: file requirements, SSE streaming latency, 7-dimension rule scores, and health statuses."
        />
      </Head>

      <div className="min-h-screen flex flex-col bg-ground text-body">
        <Header />

        <main
          ref={mainRef}
          id="main-content"
          className="flex-1 pt-[84px] sm:pt-[108px] pb-16"
        >
          {/* Intro */}
          <section className="pb-12 sm:pb-16">
            <div className="wrap space-y-5">
              <p
                data-help-hero
                className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-soft border border-accent-line font-mono text-[13px] text-accent-text"
              >
                Documentation & troubleshooting
              </p>
              <h1
                data-help-hero
                className="text-[34px] sm:text-[46px] lg:text-[52px] leading-[1.08] text-ink max-w-3xl"
              >
                Usage guide and{' '}
                <span className="text-gradient">common questions</span>
              </h1>
              <p
                data-help-hero
                className="text-[17px] leading-[1.55] text-body max-w-[58ch]"
              >
                Everything needed to prepare a resume, interpret the seven rule
                weights and literal quotes, and diagnose local FastAPI or Ollama
                health states.
              </p>
              <div data-help-hero className="pt-1">
                <Link href="/analyze" className="btn-primary w-full sm:w-auto">
                  <span>Upload a CV</span>
                  <ArrowRightIcon
                    className="w-4 h-4 m-arrow"
                    aria-hidden="true"
                  />
                </Link>
              </div>
            </div>
          </section>

          {/* Quick Reference Bento Strip */}
          <section
            aria-label="Quick specifications"
            className="py-12 sm:py-16 border-t border-hairline"
          >
            <div className="wrap">
              <div
                data-spec-grid
                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
              >
                {QUICK_SPECS.map((spec) => (
                  <div
                    key={spec.label}
                    data-spec-card
                    className={`bento-card ${
                      spec.highlight ? 'bento-card--highlight' : ''
                    } p-6 flex flex-col justify-between gap-4`}
                  >
                    <div className="space-y-1">
                      <p className="text-[13px] text-muted">{spec.label}</p>
                      <p className="text-[24px] font-mono font-medium tracking-[-0.03em] text-ink">
                        {spec.value}
                      </p>
                    </div>
                    <p className="text-[14px] text-body leading-relaxed">
                      {spec.detail}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Open FAQ Bento Grid (all answers visible at rest) */}
          <section
            id="faq"
            aria-labelledby="faq-heading"
            className="py-12 sm:py-16 border-t border-hairline"
          >
            <div className="wrap space-y-10">
              <div className="max-w-2xl space-y-2">
                <h2
                  id="faq-heading"
                  className="text-[26px] sm:text-[36px] leading-[1.14] text-ink"
                >
                  Common questions
                </h2>
                <p className="text-[16px] text-body">
                  All answers are shown directly so you can scan requirements
                  and scoring behavior in one read.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {FAQS.map((faq) => (
                  <article
                    key={faq.question}
                    data-faq-card
                    className="bento-card p-6 sm:p-8 space-y-3"
                  >
                    <h3 className="text-[18px] sm:text-[19px] leading-snug text-ink">
                      {faq.question}
                    </h3>
                    <p className="text-[15px] text-body leading-relaxed">
                      {faq.answer}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          </section>
        </main>

        <Footer />
      </div>
    </>
  );
};

export default HelpPage;
