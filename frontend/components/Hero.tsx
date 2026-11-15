import React, { useRef, useId } from 'react';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import {
  gsap,
  MOTION_QUERIES,
  SplitText,
  useGSAP,
} from '../utils/gsap';

const PREVIEW_AXES = [
  { name: 'Skills', weight: 30, score: 46, scored: true },
  { name: 'Title', weight: 20, score: 50, scored: false },
  { name: 'Experience', weight: 15, score: 50, scored: false },
  { name: 'Education', weight: 10, score: 100, scored: true },
  { name: 'Industry', weight: 10, score: 100, scored: true },
  { name: 'Location', weight: 8, score: 50, scored: false },
  { name: 'Salary', weight: 7, score: 75, scored: false },
];

const HERO_FIT_SCORE = 99.5;
const HERO_RING_RADIUS = 20;
const HERO_RING_CIRCUMFERENCE = 2 * Math.PI * HERO_RING_RADIUS;
const HERO_RING_OFFSET =
  HERO_RING_CIRCUMFERENCE - (HERO_FIT_SCORE / 100) * HERO_RING_CIRCUMFERENCE;

const Hero: React.FC = () => {
  const heroRef = useRef<HTMLElement | null>(null);
  const glowRef = useRef<HTMLDivElement | null>(null);
  const plainHeadlineRef = useRef<HTMLSpanElement | null>(null);
  const gradHeadlineRef = useRef<HTMLSpanElement | null>(null);
  const corpusNumRef = useRef<HTMLSpanElement | null>(null);
  const weightsNumRef = useRef<HTMLSpanElement | null>(null);
  const repeatNumRef = useRef<HTMLSpanElement | null>(null);
  const ringArcRef = useRef<SVGCircleElement | null>(null);
  const ringNumRef = useRef<HTMLSpanElement | null>(null);

  const rawId = useId();
  const ringGradId = `hero-fit-grad-${rawId.replace(/:/g, '')}`;

  useGSAP(
    () => {
      if (!heroRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          const { canAnimate } = context.conditions ?? {};
          if (!canAnimate) return;

          const split = plainHeadlineRef.current
            ? SplitText.create(plainHeadlineRef.current, {
                type: 'words',
                tag: 'span',
                wordsClass: 'inline-block',
                aria: 'none',
              })
            : null;

          const headlineUnits = [
            ...(split ? split.words : []),
            ...(gradHeadlineRef.current ? [gradHeadlineRef.current] : []),
          ];

          const tl = gsap.timeline({
            defaults: { ease: 'power3.out' },
          });

          tl.addLabel('start', 0)
            .from(
              '[data-hero-eyebrow]',
              {
                y: 12,
                opacity: 0,
                duration: 0.42,
                clearProps: 'transform,opacity',
              },
              'start'
            )
            .from(
              headlineUnits,
              {
                yPercent: 28,
                opacity: 0,
                duration: 0.54,
                stagger: 0.05,
                clearProps: 'transform,opacity',
              },
              'start+=0.05'
            )
            .from(
              '[data-hero-sub]',
              {
                y: 14,
                opacity: 0,
                duration: 0.46,
                clearProps: 'transform,opacity',
              },
              '-=0.3'
            )
            .from(
              '[data-hero-cta]',
              {
                y: 12,
                opacity: 0,
                duration: 0.42,
                stagger: 0.075,
                clearProps: 'transform,opacity',
              },
              '-=0.28'
            )
            .from(
              '[data-hero-stat]',
              {
                y: 12,
                opacity: 0,
                duration: 0.4,
                stagger: 0.065,
                clearProps: 'transform,opacity',
              },
              '-=0.22'
            );

          // Live number counters for real facts (resting DOM holds final text)
          const snapInt = gsap.utils.snap(1);
          const corpusState = { val: 0 };
          const weightsState = { val: 0 };
          const repeatState = { val: 0 };

          if (corpusNumRef.current) {
            tl.to(
              corpusState,
              {
                val: 361,
                duration: 0.95,
                ease: 'power2.out',
                onUpdate: () => {
                  if (corpusNumRef.current) {
                    corpusNumRef.current.textContent = String(
                      snapInt(corpusState.val)
                    );
                  }
                },
              },
              'start+=0.32'
            );
          }

          if (weightsNumRef.current) {
            tl.to(
              weightsState,
              {
                val: 100,
                duration: 0.85,
                ease: 'power2.out',
                onUpdate: () => {
                  if (weightsNumRef.current) {
                    weightsNumRef.current.textContent = String(
                      snapInt(weightsState.val)
                    );
                  }
                },
              },
              'start+=0.38'
            );
          }

          if (repeatNumRef.current) {
            tl.to(
              repeatState,
              {
                val: 0.9,
                duration: 0.85,
                ease: 'power2.out',
                onUpdate: () => {
                  if (repeatNumRef.current) {
                    repeatNumRef.current.textContent =
                      repeatState.val.toFixed(1);
                  }
                },
              },
              'start+=0.44'
            );
          }

          // Workspace preview assembling sequence
          tl.from(
            '[data-hero-stage]',
            {
              y: 20,
              scale: 0.98,
              opacity: 0,
              duration: 0.6,
              clearProps: 'transform,opacity',
            },
            'start+=0.14'
          )
            .from(
              '[data-hero-frag-head]',
              {
                y: 10,
                opacity: 0,
                duration: 0.36,
                stagger: 0.06,
                clearProps: 'transform,opacity',
              },
              '-=0.34'
            )
            .from(
              '[data-hero-bar]',
              {
                scaleX: 0,
                transformOrigin: 'left center',
                duration: 0.56,
                stagger: 0.055,
                ease: 'power2.out',
                clearProps: 'transform',
              },
              '-=0.22'
            )
            .from(
              '[data-hero-chip]',
              {
                scale: 0.92,
                opacity: 0,
                duration: 0.28,
                stagger: 0.03,
                clearProps: 'transform,opacity',
              },
              '-=0.26'
            )
            .from(
              '[data-hero-quote]',
              {
                y: 8,
                opacity: 0,
                duration: 0.36,
                clearProps: 'transform,opacity',
              },
              '-=0.16'
            );

          if (ringArcRef.current) {
            tl.fromTo(
              ringArcRef.current,
              { strokeDashoffset: HERO_RING_CIRCUMFERENCE },
              {
                strokeDashoffset: HERO_RING_OFFSET,
                duration: 0.9,
                ease: 'power3.out',
              },
              'start+=0.36'
            );
          }

          if (ringNumRef.current) {
            const fitState = { val: 0 };
            tl.to(
              fitState,
              {
                val: HERO_FIT_SCORE,
                duration: 0.9,
                ease: 'power2.out',
                onUpdate: () => {
                  if (ringNumRef.current) {
                    ringNumRef.current.textContent = fitState.val.toFixed(1);
                  }
                },
              },
              'start+=0.36'
            );
          }

          // Subtle scroll-linked parallax on the ambient glow layer (transform only)
          if (glowRef.current) {
            gsap.to(glowRef.current, {
              yPercent: 18,
              scale: 1.05,
              ease: 'none',
              scrollTrigger: {
                trigger: heroRef.current,
                start: 'top top',
                end: 'bottom top',
                scrub: 0.8,
              },
            });
          }

          return () => {
            split?.revert();
            if (corpusNumRef.current) corpusNumRef.current.textContent = '361';
            if (weightsNumRef.current)
              weightsNumRef.current.textContent = '100';
            if (repeatNumRef.current) repeatNumRef.current.textContent = '0.9';
            if (ringNumRef.current) ringNumRef.current.textContent = '99.5';
            if (ringArcRef.current) {
              ringArcRef.current.style.strokeDashoffset =
                String(HERO_RING_OFFSET);
            }
          };
        },
        heroRef
      );

      // Separate desktop-only pointer drift so viewport resizes never replay the hero entrance
      mm.add(
        '(prefers-reduced-motion: no-preference) and (min-width: 1024px) and (hover: hover) and (pointer: fine)',
        () => {
          if (!glowRef.current || !heroRef.current) return;
          const clampShift = gsap.utils.clamp(-20, 20);
          const xTo = gsap.quickTo(glowRef.current, 'x', {
            duration: 0.65,
            ease: 'power3.out',
          });
          const yTo = gsap.quickTo(glowRef.current, 'y', {
            duration: 0.65,
            ease: 'power3.out',
          });

          const onPointerMove = (e: PointerEvent) => {
            const rect = heroRef.current?.getBoundingClientRect();
            if (!rect || rect.width === 0 || rect.height === 0) return;
            const nx = (e.clientX - rect.left) / rect.width - 0.5;
            const ny = (e.clientY - rect.top) / rect.height - 0.5;
            xTo(clampShift(nx * 36));
            yTo(clampShift(ny * 28));
          };

          const el = heroRef.current;
          el.addEventListener('pointermove', onPointerMove, { passive: true });
          return () => {
            el.removeEventListener('pointermove', onPointerMove);
          };
        },
        heroRef
      );

      return () => mm.revert();
    },
    { scope: heroRef }
  );

  return (
    <section
      ref={heroRef}
      aria-labelledby="hero-heading"
      className="relative isolate overflow-hidden pt-[80px] sm:pt-[104px] pb-12 lg:pb-16 lg:min-h-[100svh] flex items-center"
    >
      {/* Ambient Brand Gradient Glow */}
      <div
        ref={glowRef}
        aria-hidden="true"
        className="hero-glow absolute -inset-x-8 -top-12 -bottom-12 -z-10"
      />

      <div className="wrap">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-6 items-start">
          {/* Left Column: Plain Category Headline, Hard Facts, Two Actions */}
          <div className="lg:col-span-5 space-y-6 pt-1 min-w-0">
            <p
              data-hero-eyebrow
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-soft border border-accent-line text-[13px] font-mono text-accent-text"
            >
              <span
                aria-hidden="true"
                className="w-1.5 h-1.5 rounded-full bg-accent"
              />
              <span>361 CareerViet postings · 12&nbsp;GB local GPU</span>
            </p>

            <h1
              id="hero-heading"
              className="text-[31px] sm:text-[46px] lg:text-[52px] leading-[1.06] text-ink"
            >
              <span ref={plainHeadlineRef}>CV matching for</span>{' '}
              <span ref={gradHeadlineRef} className="text-gradient inline-block">
                CareerViet job seekers
              </span>
            </h1>

            <p
              data-hero-sub
              className="text-[16px] sm:text-[18px] leading-[1.55] text-body max-w-[44ch]"
            >
              Upload a PDF, DOCX, or TXT resume once. Fine-tuned Qwen2.5-7B
              extracts your history, retrieves 20 candidate roles, and scores
              each posting across seven weighted rules.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 pt-1">
              <Link
                href="/analyze"
                data-hero-cta
                className="btn-primary w-full sm:w-auto"
              >
                <span>Upload a CV</span>
                <ArrowRightIcon
                  className="w-4 h-4 m-arrow"
                  aria-hidden="true"
                />
              </Link>
              <Link
                href="/about"
                data-hero-cta
                className="btn-secondary w-full sm:w-auto"
              >
                <span>Read architecture</span>
              </Link>
            </div>

            {/* Docked Fact Ledger */}
            <dl className="pt-6 border-t border-hairline grid grid-cols-3 gap-3 sm:gap-4">
              <div data-hero-stat>
                <dt className="text-[13px] text-muted">Corpus</dt>
                <dd className="text-[20px] sm:text-[26px] font-medium tracking-[-0.03em] text-ink font-mono tabular-nums mt-0.5">
                  <span ref={corpusNumRef}>361</span>
                </dd>
              </div>
              <div data-hero-stat>
                <dt className="text-[13px] text-muted">Rule weights</dt>
                <dd className="text-[20px] sm:text-[26px] font-medium tracking-[-0.03em] text-ink font-mono tabular-nums mt-0.5">
                  <span ref={weightsNumRef}>100</span>&nbsp;pt
                </dd>
              </div>
              <div data-hero-stat>
                <dt className="text-[13px] text-muted">Repeat CV</dt>
                <dd className="text-[20px] sm:text-[26px] font-medium tracking-[-0.03em] text-ink font-mono tabular-nums mt-0.5">
                  ~<span ref={repeatNumRef}>0.9</span>&nbsp;s
                </dd>
              </div>
            </dl>
          </div>

          {/* Right Column: Cobalt Bento Stage Holding Live HTML Matcher Window */}
          <div className="lg:col-span-7 min-w-0">
            <div
              data-hero-stage
              data-visual="matcher-window"
              className="bento-panel bento-panel--cobalt p-4 sm:p-6"
            >
              <div className="ui-frag p-4 sm:p-6 space-y-4">
                {/* Window Header */}
                <div
                  data-hero-frag-head
                  className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-hairline text-[13px] font-mono text-muted"
                >
                  <span>CV/3.txt · run #3 · pool 20 of 361</span>
                  <span className="text-accent-text font-medium">
                    hybrid_qwen_v1
                  </span>
                </div>

                {/* Top Ranked Job Header + Brand Gradient Fit Ring */}
                <div
                  data-hero-frag-head
                  className="flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[13px] font-mono text-muted">
                      <span className="text-ink font-medium">#1</span>
                      <span>jd_156</span>
                      <span aria-hidden="true">·</span>
                      <span>search&nbsp;#1</span>
                      <span aria-hidden="true">·</span>
                      <span>rule&nbsp;60.4/100</span>
                    </div>
                    <p className="text-[19px] sm:text-[21px] font-medium leading-snug text-ink mt-0.5">
                      Full Stack Developer
                    </p>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span className="inline-flex items-center px-2.5 py-1 rounded-ctrl border border-good-line bg-good-bg text-[13px] font-medium text-good">
                      Good Fit
                    </span>
                    <div
                      className="relative w-12 h-12 flex items-center justify-center shrink-0"
                      aria-label="Overall fit score 99.5 out of 100"
                    >
                      <svg
                        className="w-12 h-12 -rotate-90"
                        viewBox="0 0 48 48"
                        aria-hidden="true"
                      >
                        <defs>
                          <linearGradient
                            id={ringGradId}
                            x1="0%"
                            y1="0%"
                            x2="100%"
                            y2="100%"
                          >
                            <stop offset="0%" stopColor="var(--ring-stop-1)" />
                            <stop offset="50%" stopColor="var(--ring-stop-2)" />
                            <stop
                              offset="100%"
                              stopColor="var(--ring-stop-3)"
                            />
                          </linearGradient>
                        </defs>
                        <circle
                          cx="24"
                          cy="24"
                          r={HERO_RING_RADIUS}
                          fill="none"
                          stroke="var(--line-strong)"
                          strokeWidth="4"
                        />
                        <circle
                          ref={ringArcRef}
                          cx="24"
                          cy="24"
                          r={HERO_RING_RADIUS}
                          fill="none"
                          stroke={`url(#${ringGradId})`}
                          strokeWidth="4"
                          strokeLinecap="round"
                          strokeDasharray={HERO_RING_CIRCUMFERENCE}
                          strokeDashoffset={HERO_RING_OFFSET}
                        />
                      </svg>
                      <span
                        ref={ringNumRef}
                        className="absolute font-mono text-[11px] font-medium text-ink tabular-nums"
                      >
                        99.5
                      </span>
                    </div>
                  </div>
                </div>

                {/* Weighted 7-Axis Breakdown */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-5 gap-y-2 pt-1">
                  {PREVIEW_AXES.map((axis) => (
                    <div
                      key={axis.name}
                      className="grid grid-cols-[84px_1fr_34px] sm:grid-cols-[96px_1fr_38px] items-center gap-2 sm:gap-2.5 text-[13px]"
                    >
                      <span
                        className={
                          axis.scored ? 'text-ink font-medium' : 'text-muted'
                        }
                      >
                        {axis.name}{' '}
                        <span className="font-mono text-muted">
                          {axis.weight}
                        </span>
                      </span>
                      <div className="h-1.5 rounded-full bg-surface overflow-hidden">
                        {axis.scored ? (
                          <div
                            data-hero-bar
                            className="h-full rounded-full bar-brand origin-left"
                            style={{ width: `${axis.score}%` }}
                          />
                        ) : (
                          <div className="w-full h-full border-t border-dashed border-muted/60 mt-[2px]" />
                        )}
                      </div>
                      <span className="font-mono text-right tabular-nums text-ink">
                        {axis.scored ? axis.score : '—'}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Matched & Missing Skills */}
                <div className="pt-3 border-t border-hairline flex flex-wrap items-center gap-1.5 text-[13px]">
                  <span className="font-medium text-muted mr-1">Matched:</span>
                  {['typescript', 'angular', 'java', 'sql', 'git'].map((s) => (
                    <span
                      key={s}
                      data-hero-chip
                      className="px-2 py-0.5 rounded-[6px] border border-good-line bg-good-bg text-good font-mono"
                    >
                      {s}
                    </span>
                  ))}
                  <span className="font-medium text-muted ml-1 mr-1">
                    Missing:
                  </span>
                  {['docker', 'redis'].map((s) => (
                    <span
                      key={s}
                      data-hero-chip
                      className="px-2 py-0.5 rounded-[6px] border border-pot-line bg-pot-bg text-pot font-mono"
                    >
                      {s}
                    </span>
                  ))}
                </div>

                {/* Literal Quote Evidence */}
                <div
                  data-hero-quote
                  className="rounded-ctrl bg-surface px-3.5 py-2.5 border border-hairline text-[13px] font-mono text-body"
                >
                  <span className="text-muted mr-2">CV quote:</span>
                  <span>
                    “Developing backend website application. RESTful web service
                    based on Spring Boot.”
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
