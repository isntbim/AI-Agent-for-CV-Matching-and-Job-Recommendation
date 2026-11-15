import React, { useState, useMemo, useRef, useEffect } from 'react';
import Head from 'next/head';
import {
  ArrowRightIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import Header from '../components/Header';
import Footer from '../components/Footer';
import FileUpload from '../components/FileUpload';
import AnalysisResults from '../components/AnalysisResults';
import { useAnalysis } from '../hooks/useAnalysis';
import { PipelineStep, todayIsoDate } from '../services/api';
import { gsap, MOTION_QUERIES, useGSAP } from '../utils/gsap';

type InputMode = 'upload' | 'paste';

const PIPELINE_STEPS: { key: PipelineStep; label: string; detail: string }[] = [
  {
    key: 'extract',
    label: 'Reading the CV',
    detail: 'QLoRA Qwen2.5-7B structured extraction and quote verification',
  },
  {
    key: 'retrieve',
    label: 'Searching 361 jobs',
    detail: 'Hybrid BGE-M3 dense vectors and BM25 lexical fusion (top 20)',
  },
  {
    key: 'score',
    label: 'Scoring each job',
    detail: 'Deterministic V3.1 rules across seven weighted axes',
  },
  {
    key: 'rank',
    label: 'Ranking candidates',
    detail: 'hybrid_qwen_v1 logistic blend with Qwen3-Embedding-4B cosines',
  },
];

const STEP_PROGRESS: Record<PipelineStep, number> = {
  extract: 0.24,
  retrieve: 0.52,
  score: 0.78,
  rank: 0.94,
};

function formatRunTimestamp(iso: string): string {
  if (!iso) return '';
  return iso.replace('T', ' ').slice(0, 16);
}

const AnalyzePage: React.FC = () => {
  const [inputMode, setInputMode] = useState<InputMode>('upload');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [pastedText, setPastedText] = useState('');
  const [topJobs, setTopJobs] = useState<number>(5);
  const [asOf, setAsOf] = useState<string>(() => todayIsoDate());

  const pageRef = useRef<HTMLElement | null>(null);
  const streamCardRef = useRef<HTMLDivElement | null>(null);
  const streamBarRef = useRef<HTMLDivElement | null>(null);
  const tokenPreRef = useRef<HTMLPreElement | null>(null);

  const {
    isLoading,
    isLoadingHistoryRun,
    elapsedSeconds,
    activeStep,
    completedSteps,
    streamTokens,
    streamTail,
    results,
    history,
    error,
    runMatch,
    loadRun,
    refreshHistory,
    reset,
  } = useAnalysis();

  const trimmedLength = pastedText.trim().length;
  const canSubmit =
    inputMode === 'upload' ? Boolean(selectedFile) : trimmedLength >= 200;

  const completedCount = Object.keys(completedSteps).length;
  const streamProgressRatio = useMemo(() => {
    const clampRatio = gsap.utils.clamp(0.08, 1);
    if (activeStep && STEP_PROGRESS[activeStep]) {
      return clampRatio(STEP_PROGRESS[activeStep]);
    }
    if (completedCount > 0) {
      return clampRatio((completedCount / PIPELINE_STEPS.length) * 0.96);
    }
    return 0.12;
  }, [activeStep, completedCount]);

  // Initial page reveal (subtle, once on mount)
  useGSAP(
    () => {
      if (!pageRef.current) return;
      const mm = gsap.matchMedia();
      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
          tl.from('[data-analyze-head]', {
            y: 12,
            opacity: 0,
            duration: 0.42,
            clearProps: 'transform,opacity',
          })
            .from(
              '[data-analyze-panel]',
              {
                y: 14,
                opacity: 0.88,
                duration: 0.44,
                stagger: 0.07,
                clearProps: 'transform,opacity',
              },
              '-=0.26'
            )
            .from(
              '[data-idle-tile]',
              {
                y: 12,
                opacity: 0.88,
                duration: 0.38,
                stagger: 0.05,
                clearProps: 'transform,opacity',
              },
              '-=0.28'
            );
        },
        pageRef
      );
      return () => mm.revert();
    },
    { scope: pageRef }
  );

  // Streaming card mount entrance (runs once each time streaming starts)
  useGSAP(
    () => {
      if (!isLoading || !streamCardRef.current) return;
      const mm = gsap.matchMedia();
      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          gsap.from('[data-stream-step]', {
            y: 10,
            opacity: 0.85,
            duration: 0.34,
            stagger: 0.05,
            ease: 'power2.out',
            clearProps: 'transform,opacity',
          });
        },
        streamCardRef
      );
      return () => mm.revert();
    },
    { scope: streamCardRef, dependencies: [isLoading], revertOnUpdate: true }
  );

  // Keep live streamed JSON preview scrolled to the latest generated token
  useEffect(() => {
    if (tokenPreRef.current && streamTail) {
      tokenPreRef.current.scrollTop = tokenPreRef.current.scrollHeight;
    }
  }, [streamTail]);

  // Active pipeline step pulse & progress bar smoothing during streaming
  useGSAP(
    () => {
      if (!isLoading || !streamCardRef.current) return;
      const mm = gsap.matchMedia();
      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          if (streamBarRef.current) {
            gsap.to(streamBarRef.current, {
              scaleX: streamProgressRatio,
              transformOrigin: 'left center',
              duration: 0.45,
              ease: 'power2.out',
            });
          }
          gsap.to('[data-step-pulse="true"]', {
            scale: 1.45,
            opacity: 0.35,
            duration: 0.65,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut',
          });
        },
        streamCardRef
      );
      return () => mm.revert();
    },
    {
      scope: streamCardRef,
      dependencies: [isLoading, activeStep, streamProgressRatio],
      revertOnUpdate: true,
    }
  );

  // Discrete screen-reader announcement only on stage changes or completion (excludes timer ticks & token tail)
  const statusAnnouncement = useMemo(() => {
    if (isLoading && activeStep) {
      const idx = PIPELINE_STEPS.findIndex((s) => s.key === activeStep);
      const step = PIPELINE_STEPS[idx];
      if (step) {
        return `Stage ${idx + 1} of ${PIPELINE_STEPS.length}: ${step.label}.`;
      }
    }
    if (!isLoading && results) {
      return `Ranked ${results.matches.length} of ${results.pool} candidate jobs in ${results.seconds.toFixed(1)} seconds.`;
    }
    return '';
  }, [isLoading, activeStep, results]);

  const switchMode = (nextMode: InputMode) => {
    setInputMode(nextMode);
    if (nextMode === 'upload') {
      setPastedText('');
    } else {
      setSelectedFile(null);
    }
  };

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    if (file) {
      setPastedText('');
    }
  };

  const scrollResultsOnMobile = () => {
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      const reduce = window.matchMedia(
        '(prefers-reduced-motion: reduce)'
      ).matches;
      const panel = document.getElementById('results-panel');
      panel?.scrollIntoView({
        behavior: reduce ? 'auto' : 'smooth',
        block: 'start',
      });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const ok = await runMatch({
      file: inputMode === 'upload' ? selectedFile : null,
      text: inputMode === 'paste' ? pastedText : undefined,
      top: topJobs,
      asOf: asOf || undefined,
    });
    if (ok) {
      scrollResultsOnMobile();
    }
  };

  const handleSelectHistoryRun = async (runId: number) => {
    const ok = await loadRun(runId);
    if (ok) {
      scrollResultsOnMobile();
    }
  };

  const handleNewAnalysis = () => {
    reset();
    setSelectedFile(null);
    setPastedText('');
  };

  return (
    <>
      <Head>
        <title>Upload a CV — ResuMatch</title>
        <meta
          name="description"
          content="Upload a PDF, DOCX, or TXT CV or paste resume text to rank top CareerViet job matches with 7-dimension rule scores and verbatim quotes."
        />
      </Head>

      <div className="min-h-screen flex flex-col bg-ground text-body">
        <Header />

        <main
          ref={pageRef}
          id="main-content"
          className="flex-1 pt-[80px] sm:pt-[100px] pb-16"
        >
          <div className="wrap space-y-8">
            {/* Workspace Header */}
            <header data-analyze-head className="space-y-2 max-w-3xl">
              <h1 className="text-[30px] sm:text-[38px] leading-[1.1] text-ink">
                Match a CV against{' '}
                <span className="text-gradient">361 CareerViet jobs</span>
              </h1>
              <p className="text-[16px] text-body leading-relaxed">
                Upload a resume file or paste plain text. First-time extraction
                streams live from Qwen2.5-7B in 20–60&nbsp;s; cached CVs return
                in under a second.
              </p>
            </header>

            {/* Discrete polite status region (stage transitions & completion only; no timer or JSON stream spam) */}
            <div
              role="status"
              aria-live="polite"
              aria-atomic="true"
              className="sr-only"
            >
              {statusAnnouncement}
            </div>

            {/* 12-Column Bento Workspace */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Input Form + Recent Runs */}
              <div className="lg:col-span-4 space-y-6 min-w-0">
                <form
                  data-analyze-panel
                  onSubmit={handleSubmit}
                  className="bento-card p-5 sm:p-6 space-y-5"
                  noValidate
                >
                  {/* Segmented Input Mode Switch */}
                  <div className="space-y-2">
                    <span className="block text-[14px] font-medium text-ink">
                      Resume source
                    </span>
                    <div
                      role="group"
                      aria-label="Resume input method"
                      className="grid grid-cols-2 gap-1.5 p-1 rounded-ctrl bg-ground border border-hairline"
                    >
                      <button
                        type="button"
                        disabled={isLoading}
                        aria-pressed={inputMode === 'upload'}
                        onClick={() => switchMode('upload')}
                        className={`min-h-[44px] px-3 rounded-[6px] text-[14px] font-medium transition-colors ${
                          inputMode === 'upload'
                            ? 'bg-card text-ink border border-hairline-strong'
                            : 'text-muted hover:text-ink border border-transparent'
                        }`}
                      >
                        Upload file
                      </button>
                      <button
                        type="button"
                        disabled={isLoading}
                        aria-pressed={inputMode === 'paste'}
                        onClick={() => switchMode('paste')}
                        className={`min-h-[44px] px-3 rounded-[6px] text-[14px] font-medium transition-colors ${
                          inputMode === 'paste'
                            ? 'bg-card text-ink border border-hairline-strong'
                            : 'text-muted hover:text-ink border border-transparent'
                        }`}
                      >
                        Paste CV text
                      </button>
                    </div>
                  </div>

                  {/* Active Input Control */}
                  {inputMode === 'upload' ? (
                    <FileUpload
                      selectedFile={selectedFile}
                      onFileSelect={handleFileSelect}
                      disabled={isLoading}
                    />
                  ) : (
                    <div className="space-y-2">
                      <div className="flex items-baseline justify-between gap-2">
                        <label
                          htmlFor="cv-text-input"
                          className="text-[14px] font-medium text-ink"
                        >
                          CV text
                        </label>
                        <span
                          className={`font-mono text-[12px] tabular-nums ${
                            trimmedLength >= 200 ? 'text-good' : 'text-muted'
                          }`}
                        >
                          {trimmedLength} / 200 min chars
                        </span>
                      </div>
                      <textarea
                        id="cv-text-input"
                        rows={8}
                        value={pastedText}
                        disabled={isLoading}
                        onChange={(e) => setPastedText(e.target.value)}
                        placeholder="Paste work history, skills, projects, and education..."
                        className="field-input resize-y min-h-[190px]"
                      />
                    </div>
                  )}

                  {/* Options stacked inside narrow lg:col-span-4 sidebar so 1024px never clips dates or labels */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-3 pt-1">
                    <div className="space-y-1.5">
                      <label
                        htmlFor="top-jobs-select"
                        className="block text-[14px] font-medium text-ink"
                      >
                        Jobs to return
                      </label>
                      <select
                        id="top-jobs-select"
                        value={topJobs}
                        disabled={isLoading}
                        onChange={(e) => setTopJobs(Number(e.target.value))}
                        className="field-input"
                      >
                        <option value={3}>Top 3 jobs</option>
                        <option value={5}>Top 5 jobs</option>
                        <option value={10}>Top 10 jobs</option>
                      </select>
                    </div>

                    <div className="space-y-1.5">
                      <label
                        htmlFor="as-of-input"
                        className="block text-[14px] font-medium text-ink"
                      >
                        Experience up to
                      </label>
                      <input
                        id="as-of-input"
                        type="date"
                        value={asOf}
                        disabled={isLoading}
                        onChange={(e) => setAsOf(e.target.value)}
                        className="field-input font-mono text-[15px]"
                      />
                    </div>
                  </div>

                  {/* Primary Action */}
                  <button
                    type="submit"
                    disabled={!canSubmit || isLoading}
                    className="btn-primary w-full"
                  >
                    {isLoading ? (
                      <span className="font-mono tabular-nums">
                        Matching CV… ({elapsedSeconds}&nbsp;s)
                      </span>
                    ) : (
                      <>
                        <span>Run CV match</span>
                        <ArrowRightIcon
                          className="w-4 h-4 m-arrow"
                          aria-hidden="true"
                        />
                      </>
                    )}
                  </button>
                </form>

                {/* Recent Runs Bento Card (/api/history) */}
                <section
                  data-analyze-panel
                  aria-labelledby="recent-runs-heading"
                  className="bento-card p-5 sm:p-6 space-y-4"
                >
                  <div className="flex items-center justify-between gap-2">
                    <h2
                      id="recent-runs-heading"
                      className="text-[17px] font-medium text-ink"
                    >
                      Recent runs
                    </h2>
                    <span className="font-mono text-[12px] text-muted">
                      {history.length} saved
                    </span>
                  </div>

                  {history.length === 0 ? (
                    <p className="text-[14px] text-muted leading-relaxed">
                      Completed runs are stored locally in SQLite and appear
                      here for one-click reload.
                    </p>
                  ) : (
                    <ul className="space-y-2">
                      {history.map((run) => {
                        const lead = run.top?.[0];
                        const isSelected = results?.run_id === run.run_id;
                        return (
                          <li key={run.run_id}>
                            <button
                              type="button"
                              disabled={isLoading || isLoadingHistoryRun}
                              onClick={() => handleSelectHistoryRun(run.run_id)}
                              className={`w-full text-left rounded-panel border p-3.5 min-h-[56px] transition-colors disabled:opacity-55 ${
                                isSelected
                                  ? 'bg-accent-soft border-accent-line'
                                  : 'bg-ground border-hairline hover:border-hairline-strong hover:bg-surface'
                              }`}
                            >
                              <div className="flex items-baseline justify-between gap-2">
                                <span className="text-[14px] font-medium text-ink truncate">
                                  {lead?.title || lead?.jd || 'Saved match run'}
                                </span>
                                <span className="font-mono text-[12px] text-muted shrink-0 tabular-nums">
                                  #{run.run_id}
                                </span>
                              </div>
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-1 text-[12px] font-mono text-muted tabular-nums">
                                {lead?.grade && (
                                  <span className="text-ink">{lead.grade}</span>
                                )}
                                {typeof lead?.fit === 'number' && (
                                  <span>{Math.round(lead.fit)}/100</span>
                                )}
                                <span aria-hidden="true">·</span>
                                <span>{run.seconds.toFixed(1)}&nbsp;s</span>
                                {run.feedback > 0 ? (
                                  <>
                                    <span aria-hidden="true">·</span>
                                    <span className="text-good">
                                      {run.feedback} feedback
                                    </span>
                                  </>
                                ) : (
                                  run.created && (
                                    <>
                                      <span aria-hidden="true">·</span>
                                      <span>
                                        {formatRunTimestamp(run.created)}
                                      </span>
                                    </>
                                  )
                                )}
                              </div>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>
              </div>

              {/* Right Column: Live Stream / Ranked Results / Idle Spec */}
              <div
                id="results-panel"
                className="lg:col-span-8 space-y-5 min-w-0"
              >
                {/* Error Alert */}
                {error && (
                  <div
                    role="alert"
                    className="bento-card border-bad-line bg-bad-bg p-4 sm:p-5 flex items-start gap-3"
                  >
                    <ExclamationCircleIcon
                      className="w-5 h-5 text-bad shrink-0 mt-0.5"
                      aria-hidden="true"
                    />
                    <div className="space-y-1 min-w-0">
                      <p className="text-[15px] font-medium text-bad">
                        Matching could not be completed
                      </p>
                      <p className="text-[14px] text-ink [overflow-wrap:anywhere] break-words">
                        {error}
                      </p>
                    </div>
                  </div>
                )}

                {/* Live SSE Streaming Card */}
                {isLoading && (
                  <div
                    ref={streamCardRef}
                    className="bento-card bento-card--highlight p-6 sm:p-8 space-y-6"
                  >
                    <div className="space-y-4 pb-4 border-b border-hairline">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="space-y-1">
                          <h2 className="text-[21px] leading-snug text-ink">
                            Running CareerViet matching pipeline
                          </h2>
                          <p className="text-[14px] text-muted">
                            Streaming live progress over{' '}
                            <span className="font-mono">/api/match/stream</span>
                          </p>
                        </div>
                        <div
                          aria-hidden="true"
                          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-ctrl bg-ground border border-hairline font-mono text-[13px] text-ink tabular-nums"
                        >
                          <ClockIcon className="w-4 h-4 text-accent-text" />
                          <span>{elapsedSeconds}&nbsp;s</span>
                          {streamTokens > 0 && (
                            <>
                              <span className="text-muted">·</span>
                              <span
                                data-live-token-badge
                                className="text-accent-text font-medium"
                              >
                                {streamTokens} tokens
                              </span>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Smooth compositor-scaled pipeline progress bar */}
                      <div
                        aria-hidden="true"
                        className="h-1.5 w-full rounded-full bg-surface overflow-hidden"
                      >
                        <div
                          ref={streamBarRef}
                          className="h-full w-full rounded-full bar-brand origin-left"
                          style={{ transform: `scaleX(${streamProgressRatio})` }}
                        />
                      </div>
                    </div>

                    {/* 4-Stage Checklist */}
                    <ol className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {PIPELINE_STEPS.map((step, idx) => {
                        const doneTime = completedSteps[step.key];
                        const isDone = typeof doneTime === 'number';
                        const isCurrent = activeStep === step.key;

                        return (
                          <li
                            key={step.key}
                            data-stream-step
                            className={`rounded-panel border p-4 space-y-1.5 transition-colors ${
                              isDone
                                ? 'bg-good-bg/60 border-good-line'
                                : isCurrent
                                  ? 'bg-accent-soft border-accent-line'
                                  : 'bg-surface border-hairline border-dashed'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-mono text-[12px] text-muted">
                                0{idx + 1}
                              </span>
                              {isDone ? (
                                <span className="inline-flex items-center gap-1 font-mono text-[12px] text-good tabular-nums">
                                  <CheckCircleIcon
                                    className="w-4 h-4"
                                    aria-hidden="true"
                                  />
                                  <span>{doneTime.toFixed(2)}&nbsp;s</span>
                                </span>
                              ) : isCurrent ? (
                                <span className="inline-flex items-center gap-1.5 font-mono text-[12px] text-accent-text font-medium">
                                  <span
                                    aria-hidden="true"
                                    data-step-pulse="true"
                                    className="w-2 h-2 rounded-full bg-accent inline-block"
                                  />
                                  <span>In progress…</span>
                                </span>
                              ) : (
                                <span className="font-mono text-[12px] text-muted">
                                  Queued
                                </span>
                              )}
                            </div>
                            <p className="text-[15px] font-medium text-ink">
                              {step.label}
                            </p>
                            <p className="text-[13px] text-muted leading-relaxed">
                              {step.detail}
                            </p>
                          </li>
                        );
                      })}
                    </ol>

                    {/* Visual-only Live Extractor Token Counter & Streamed JSON Preview */}
                    {streamTail && (
                      <div
                        aria-hidden="true"
                        className="rounded-panel border border-accent-line bg-ground overflow-hidden"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-accent-soft/75 border-b border-accent-line text-[12px] font-mono">
                          <span className="inline-flex items-center gap-2 text-accent-text font-medium">
                            <span
                              data-step-pulse="true"
                              className="w-2 h-2 rounded-full bg-accent inline-block"
                            />
                            <span>qwen-v4 live structured JSON preview</span>
                          </span>
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-card border border-accent-line text-accent-text font-medium tabular-nums">
                            <span data-live-token-count>{streamTokens}</span>
                            <span>tokens</span>
                          </span>
                        </div>
                        <pre
                          ref={tokenPreRef}
                          data-live-token-tail
                          className="p-4 font-mono text-[12.5px] leading-relaxed text-ink max-h-56 overflow-y-auto whitespace-pre-wrap [overflow-wrap:anywhere] m-0"
                        >
                          {streamTail}
                        </pre>
                      </div>
                    )}
                  </div>
                )}

                {/* Ranked Results */}
                {!isLoading && results && (
                  <AnalysisResults
                    result={results}
                    onNewAnalysis={handleNewAnalysis}
                    onFeedbackSaved={refreshHistory}
                  />
                )}

                {/* Idle Bento Workspace State */}
                {!isLoading && !results && (
                  <div
                    data-analyze-panel
                    className="bento-card bento-card--highlight p-6 sm:p-8 space-y-6"
                  >
                    <div className="space-y-2 max-w-2xl">
                      <p className="font-mono text-[13px] text-accent-text">
                        361 CareerViet postings · hybrid_qwen_v1
                      </p>
                      <h2 className="text-[24px] sm:text-[28px] leading-[1.18] text-ink">
                        Ready to score a resume
                      </h2>
                      <p className="text-[16px] text-body leading-relaxed">
                        Attach a resume file or paste plain text on the left, or
                        reopen any saved run from Recent runs. Each match card
                        reports its fit score, seven rule axes, skill gaps, and
                        verbatim quotes.
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div
                        data-idle-tile
                        className="rounded-panel bg-ground border border-hairline p-4 space-y-1.5"
                      >
                        <p className="font-mono text-[12px] text-accent-text">
                          01
                        </p>
                        <h3 className="text-[16px] font-medium text-ink">
                          Quote-verified extraction
                        </h3>
                        <p className="text-[14px] text-muted leading-relaxed">
                          Qwen2.5-7B parses work history, skills, and degrees;
                          LangGraph repair rejects paraphrased quotes.
                        </p>
                      </div>

                      <div
                        data-idle-tile
                        className="rounded-panel bg-ground border border-hairline p-4 space-y-1.5"
                      >
                        <p className="font-mono text-[12px] text-accent-text">
                          02
                        </p>
                        <h3 className="text-[16px] font-medium text-ink">
                          Top-20 hybrid retrieval
                        </h3>
                        <p className="text-[14px] text-muted leading-relaxed">
                          Multilingual BGE-M3 embeddings fuse with BM25 lexical
                          matching across 361 jobs.
                        </p>
                      </div>

                      <div
                        data-idle-tile
                        className="rounded-panel bg-ground border border-hairline p-4 space-y-1.5"
                      >
                        <p className="font-mono text-[12px] text-accent-text">
                          03
                        </p>
                        <h3 className="text-[16px] font-medium text-ink">
                          Seven rule weights
                        </h3>
                        <p className="text-[14px] text-muted leading-relaxed">
                          Skills 30 · Title 20 · Experience 15 · Education 10 ·
                          Industry 10 · Location 8 · Salary 7.
                        </p>
                      </div>

                      <div
                        data-idle-tile
                        className="rounded-panel bg-ground border border-hairline p-4 space-y-1.5"
                      >
                        <p className="font-mono text-[12px] text-accent-text">
                          04
                        </p>
                        <h3 className="text-[16px] font-medium text-ink">
                          Human feedback loop
                        </h3>
                        <p className="text-[14px] text-muted leading-relaxed">
                          Record thumbs votes or override fit grades on any
                          returned posting; saved to local SQLite.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>

        <Footer hideClosingCta />
      </div>
    </>
  );
};

export default AnalyzePage;
