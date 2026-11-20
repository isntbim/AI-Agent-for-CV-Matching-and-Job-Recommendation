import React, { useState, useEffect, useRef, useId } from 'react';
import toast from 'react-hot-toast';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  DocumentTextIcon,
  ArrowPathIcon,
  ChatBubbleBottomCenterTextIcon,
  HandThumbUpIcon,
  HandThumbDownIcon,
} from '@heroicons/react/24/outline';
import {
  DIMENSION_WEIGHTS,
  FitGrade,
  InjectionFinding,
  JobMatch,
  MatchResponse,
  SavedFeedback,
  SCORE_DIMENSIONS,
  ScoreDimension,
  parseUnscoredDimensions,
  submitFeedback,
} from '../services/api';
import {
  gsap,
  MOTION_QUERIES,
  ScrollTrigger,
  useGSAP,
} from '../utils/gsap';

interface AnalysisResultsProps {
  result: MatchResponse;
  onNewAnalysis?: () => void;
  onFeedbackSaved?: () => void;
}

const DIMENSION_LABELS: Record<ScoreDimension, string> = {
  skills: 'Skills',
  title: 'Title',
  experience: 'Experience',
  education: 'Education',
  industry: 'Industry',
  location: 'Location',
  salary: 'Salary',
};

const INJECTION_KIND_LABELS: Record<string, string> = {
  override: 'instruction override',
  fake_system_message: 'system prompt markup',
  addresses_the_model: 'direct message to the AI',
  rating_order: 'score instruction',
  json_injection: 'injected JSON field',
};

// Exact weights and cut-offs from configs/hybrid_qwen_v1.json and scripts/ats_service.py.
const FIT_FORMULA_INTERCEPT = '-10.958603';
const FIT_FORMULA_FEATURES: {
  name: string;
  weight: string;
  detail: string;
}[] = [
  { name: 'skills', weight: '1.758144', detail: 'V3.1 skills / 100' },
  { name: 'title', weight: '0.137788', detail: 'V3.1 title / 100' },
  { name: 'experience', weight: '0.15821', detail: 'V3.1 experience / 100' },
  { name: 'education', weight: '-0.095872', detail: 'V3.1 education / 100' },
  { name: 'industry', weight: '0.83903', detail: 'V3.1 industry / 100' },
  {
    name: 'text_cos',
    weight: '12.967617',
    detail: 'Qwen3-Embedding-4B text cosine',
  },
  {
    name: 'title_cos',
    weight: '4.369925',
    detail: 'Qwen3-Embedding-4B title cosine',
  },
  { name: 'tfidf_cos', weight: '7.043981', detail: 'Corpus TF-IDF cosine' },
];

function formatRemovedSummary(items: InjectionFinding[]): string {
  const total = items.length;
  const cvCount = items.filter((item) => item.source === 'cv').length;
  const jdCount = items.filter((item) => item.source === 'jd').length;
  const lineWord = total === 1 ? '1 line was' : `${total} lines were`;

  if (cvCount > 0 && jdCount > 0) {
    return `${lineWord} removed before scoring (${cvCount} from your CV, ${jdCount} from the job description).`;
  }
  if (jdCount > 0) {
    return `${lineWord} removed from the job description before scoring.`;
  }
  return `${lineWord} removed from your CV before scoring.`;
}

function getGradeMeta(grade: FitGrade) {
  switch (grade) {
    case 'Good Fit':
      return {
        badgeClass: 'bg-good-bg text-good border-good-line',
        Icon: CheckCircleIcon,
      };
    case 'Potential Fit':
      return {
        badgeClass: 'bg-pot-bg text-pot border-pot-line',
        Icon: ExclamationTriangleIcon,
      };
    case 'No Fit':
    default:
      return {
        badgeClass: 'bg-bad-bg text-bad border-bad-line',
        Icon: XCircleIcon,
      };
  }
}

interface FitScoreRingProps {
  fit: number;
}

const FitScoreRing: React.FC<FitScoreRingProps> = ({ fit }) => {
  const clamped = Math.max(0, Math.min(100, fit));
  const rounded = Math.round(clamped);
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;
  const rawId = useId();
  const gradId = `fit-ring-${rawId.replace(/:/g, '')}`;

  return (
    <div
      className="relative w-16 h-16 flex items-center justify-center shrink-0"
      aria-label={`Overall fit score ${clamped.toFixed(1)} out of 100`}
    >
      <svg
        className="w-16 h-16 -rotate-90"
        viewBox="0 0 64 64"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--ring-stop-1)" />
            <stop offset="50%" stopColor="var(--ring-stop-2)" />
            <stop offset="100%" stopColor="var(--ring-stop-3)" />
          </linearGradient>
        </defs>
        <circle
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          stroke="var(--line-strong)"
          strokeWidth="5"
        />
        <circle
          data-fit-ring-arc
          data-circumference={circumference}
          data-target-offset={offset}
          cx="32"
          cy="32"
          r={radius}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center font-mono tabular-nums">
        <span
          data-fit-number
          data-target-fit={rounded}
          className="text-[14px] font-medium text-ink leading-none"
        >
          {rounded}
        </span>
        <span className="text-[10px] text-muted leading-none mt-0.5">/100</span>
      </div>
    </div>
  );
};

interface RadarChartProps {
  scores: Record<ScoreDimension, number>;
  unscored: Set<ScoreDimension>;
}

const SevenAxisRadar: React.FC<RadarChartProps> = ({ scores, unscored }) => {
  const radarWrapRef = useRef<HTMLDivElement | null>(null);
  const rawId = useId();
  const polyGradId = `radar-fill-${rawId.replace(/:/g, '')}`;

  const cx = 140;
  const cy = 122;
  const maxR = 68;
  const levels = [0.25, 0.5, 0.75, 1.0];
  const n = SCORE_DIMENSIONS.length;

  const angleForIndex = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;

  const pointAt = (i: number, ratio: number) => {
    const a = angleForIndex(i);
    return {
      x: cx + maxR * ratio * Math.cos(a),
      y: cy + maxR * ratio * Math.sin(a),
    };
  };

  const polygonPoints = SCORE_DIMENSIONS.map((dim, i) => {
    const isUnscored = unscored.has(dim);
    const ratio = isUnscored
      ? 0.35
      : Math.max(0.06, Math.min(1, (scores[dim] ?? 0) / 100));
    const pt = pointAt(i, ratio);
    return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
  }).join(' ');

  useGSAP(
    () => {
      if (!radarWrapRef.current) return;
      const mm = gsap.matchMedia();
      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate) return;
          gsap.from('[data-radar-poly]', {
            scale: 0.42,
            opacity: 0,
            svgOrigin: '140 122',
            duration: 0.58,
            ease: 'power3.out',
            clearProps: 'opacity',
          });
        },
        radarWrapRef
      );
      return () => mm.revert();
    },
    { scope: radarWrapRef }
  );

  return (
    <div ref={radarWrapRef} className="w-full max-w-[276px] mx-auto pt-1">
      <svg
        viewBox="0 0 280 244"
        className="w-full h-auto overflow-visible"
        role="img"
        aria-label="Seven-dimension rule score radar chart"
      >
        <defs>
          <linearGradient id={polyGradId} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--ring-stop-1)" />
            <stop offset="100%" stopColor="var(--ring-stop-3)" />
          </linearGradient>
        </defs>

        {levels.map((lvl) => {
          const ringPts = SCORE_DIMENSIONS.map((_, i) => {
            const pt = pointAt(i, lvl);
            return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
          }).join(' ');
          return (
            <polygon
              key={lvl}
              points={ringPts}
              fill="none"
              stroke="var(--line)"
              strokeWidth="1"
            />
          );
        })}

        {SCORE_DIMENSIONS.map((dim, i) => {
          const outer = pointAt(i, 1);
          const isUnscored = unscored.has(dim);
          return (
            <line
              key={dim}
              x1={cx}
              y1={cy}
              x2={outer.x}
              y2={outer.y}
              stroke="var(--line-strong)"
              strokeWidth={isUnscored ? '1' : '1.25'}
              strokeDasharray={isUnscored ? '4 4' : undefined}
            />
          );
        })}

        <polygon
          data-radar-poly
          points={polygonPoints}
          fill={`url(#${polyGradId})`}
          fillOpacity="0.22"
          stroke="var(--accent)"
          strokeWidth="2"
        />

        {SCORE_DIMENSIONS.map((dim, i) => {
          const isUnscored = unscored.has(dim);
          const ratio = isUnscored
            ? 0.35
            : Math.max(0.06, Math.min(1, (scores[dim] ?? 0) / 100));
          const dot = pointAt(i, ratio);
          const labelPt = pointAt(i, 1.25);
          const a = angleForIndex(i);
          const cos = Math.cos(a);
          const textAnchor =
            cos > 0.25 ? 'start' : cos < -0.25 ? 'end' : 'middle';

          return (
            <g key={dim}>
              <circle
                cx={dot.x}
                cy={dot.y}
                r={isUnscored ? 3.5 : 4.25}
                fill={isUnscored ? 'var(--card)' : 'var(--accent)'}
                stroke={isUnscored ? 'var(--muted)' : 'var(--card)'}
                strokeWidth="1.75"
              />
              <text
                x={labelPt.x}
                y={labelPt.y}
                textAnchor={textAnchor}
                dominantBaseline="middle"
                fill={isUnscored ? 'var(--muted)' : 'var(--ink)'}
                style={{
                  fontSize: '11px',
                  fontWeight: isUnscored ? 400 : 500,
                  fontStyle: isUnscored ? 'italic' : 'normal',
                }}
              >
                {DIMENSION_LABELS[dim]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

interface JobMatchCardProps {
  match: JobMatch;
  rank: number;
  runId?: number;
  initialFeedback?: SavedFeedback;
  onFeedbackSaved?: () => void;
}

const JobMatchCard: React.FC<JobMatchCardProps> = ({
  match,
  rank,
  runId,
  initialFeedback,
  onFeedbackSaved,
}) => {
  const [showRadar, setShowRadar] = useState(rank === 1);
  const [showJobText, setShowJobText] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);

  const [vote, setVote] = useState<1 | -1 | null>(
    initialFeedback?.vote ?? null
  );
  const [overrideGrade, setOverrideGrade] = useState<FitGrade | ''>(
    initialFeedback?.grade ?? ''
  );
  const [isSavingFeedback, setIsSavingFeedback] = useState(false);
  const [feedbackSavedFlag, setFeedbackSavedFlag] = useState(
    Boolean(initialFeedback)
  );

  useEffect(() => {
    setVote(initialFeedback?.vote ?? null);
    setOverrideGrade(initialFeedback?.grade ?? '');
    setFeedbackSavedFlag(Boolean(initialFeedback));
  }, [initialFeedback, runId, match.jd]);

  const unscored = parseUnscoredDimensions(match.explanation || []);
  const { badgeClass, Icon: GradeIcon } = getGradeMeta(match.grade);

  const explanationBullets =
    match.explanation && match.explanation.length > 1
      ? match.explanation.slice(1)
      : match.explanation || [];

  const evidenceEntries = Object.entries(match.evidence || {}).filter(
    ([, quotes]) => Array.isArray(quotes) && quotes.length > 0
  );

  const hasAnySkills =
    (match.matched_skills && match.matched_skills.length > 0) ||
    (match.missing_skills && match.missing_skills.length > 0);

  const persistFeedback = async (
    nextVote: 1 | -1,
    nextGrade: FitGrade | ''
  ) => {
    if (typeof runId !== 'number') return;
    setIsSavingFeedback(true);
    try {
      await submitFeedback({
        run_id: runId,
        jd: match.jd,
        vote: nextVote,
        grade: nextGrade || null,
      });
      setVote(nextVote);
      setOverrideGrade(nextGrade);
      setFeedbackSavedFlag(true);
      onFeedbackSaved?.();
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : 'Could not save feedback.'
      );
    } finally {
      setIsSavingFeedback(false);
    }
  };

  const handleVoteClick = (clickedVote: 1 | -1) => {
    let nextGrade = overrideGrade;
    if (clickedVote === 1 && nextGrade && nextGrade !== match.grade) {
      nextGrade = '';
    }
    if (clickedVote === -1 && nextGrade && nextGrade === match.grade) {
      nextGrade = '';
    }
    void persistFeedback(clickedVote, nextGrade);
  };

  const handleGradeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value as FitGrade | '';
    if (!selected) {
      if (vote !== null) {
        void persistFeedback(vote, '');
      } else {
        setOverrideGrade('');
      }
      return;
    }
    const impliedVote: 1 | -1 = selected === match.grade ? 1 : -1;
    void persistFeedback(impliedVote, selected);
  };

  const selectId = `verdict-${runId ?? 'live'}-${match.jd}-${rank}`;
  const isPastedJob = match.jd === 'pasted';

  return (
    <article
      data-result-card
      className={`bento-card ${
        rank === 1 ? 'bento-card--highlight' : ''
      } p-5 sm:p-6 space-y-5`}
    >
      {/* Card Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-5 border-b border-hairline">
        <div className="flex items-start gap-3.5 min-w-0">
          <span
            className={`inline-flex items-center justify-center min-w-[40px] h-[34px] px-2.5 rounded-ctrl border font-mono text-[13px] font-medium tabular-nums shrink-0 mt-0.5 ${
              rank === 1
                ? 'bg-accent-soft border-accent-line text-accent-text'
                : 'bg-ground border-hairline-strong text-ink'
            }`}
          >
            #{rank}
          </span>
          <div className="min-w-0 space-y-1.5">
            <h3 className="text-[19px] sm:text-[21px] leading-[1.25] font-medium text-ink [overflow-wrap:anywhere] break-words">
              {match.title ||
                (isPastedJob ? 'Your pasted job' : 'Untitled job posting')}
            </h3>
            <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[13px] font-mono text-muted tabular-nums">
              <span
                className={`px-2 py-0.5 rounded border ${
                  isPastedJob
                    ? 'bg-accent-soft border-accent-line text-accent-text'
                    : 'bg-ground border-hairline text-ink'
                }`}
              >
                {isPastedJob ? 'Your pasted job' : match.jd}
              </span>
              {!isPastedJob && (
                <>
                  <span>Retrieval #{match.retrieval_rank}</span>
                  <span aria-hidden="true">·</span>
                </>
              )}
              <span title="7-dimension V3.1 rule score; shown for breakdown and does not set the 0–100 fit score">
                Rule {match.rule_score.toFixed(1)}/100
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between sm:justify-end gap-3.5 shrink-0">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-ctrl text-[13px] font-medium border ${badgeClass}`}
          >
            <GradeIcon className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>{match.grade}</span>
          </span>
          <FitScoreRing fit={match.fit} />
        </div>
      </div>

      {/* Bento 2-Column Interior */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
        {/* Left: Explanation + Skills */}
        <div className="xl:col-span-7 space-y-5 min-w-0">
          {(match.summary || explanationBullets.length > 0) && (
            <div className="space-y-2.5">
              <h4 className="text-[14px] font-medium text-ink">
                Why this job matched
              </h4>
              {match.summary && (
                <div className="space-y-1">
                  <p className="text-[15px] text-body leading-relaxed [overflow-wrap:anywhere] break-words">
                    {match.summary}
                  </p>
                  <p className="text-[12px] text-muted">
                    {match.summary_source === 'o5'
                      ? 'Written by the explanation model and checked against the scores.'
                      : 'Standard summary: the text from the explanation model did not pass the fact check.'}
                  </p>
                </div>
              )}
              <ul className="space-y-2 text-[15px] text-body leading-relaxed">
                {explanationBullets.map((line, idx) => {
                  const isNotScoredLine = line.startsWith('Not scored');
                  return (
                    <li
                      key={idx}
                      className={`flex items-start gap-2.5 ${
                        isNotScoredLine ? 'text-[13px] text-muted' : ''
                      }`}
                    >
                      <span
                        aria-hidden="true"
                        className={`mt-2 w-1.5 h-1.5 rounded-full shrink-0 ${
                          isNotScoredLine ? 'bg-muted' : 'bg-accent'
                        }`}
                      />
                      <span className="[overflow-wrap:anywhere] break-words">
                        {line}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {/* Skill Alignment Bento Split */}
          <div className="space-y-2.5">
            <h4 className="text-[14px] font-medium text-ink">
              Skill alignment
            </h4>
            {!hasAnySkills ? (
              <p className="text-[14px] text-muted">
                No explicit skill overlap recorded on this posting.
              </p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="rounded-panel bg-ground border border-hairline p-3.5 space-y-2.5 min-w-0">
                  <div className="flex items-center justify-between gap-2 text-[13px]">
                    <span className="font-medium text-ink">Matched skills</span>
                    <span className="font-mono text-[12px] px-1.5 py-0.5 rounded bg-good-bg text-good border border-good-line tabular-nums">
                      {match.matched_skills.length}
                    </span>
                  </div>
                  {match.matched_skills.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {match.matched_skills.map((skill, idx) => (
                        <span
                          key={idx}
                          className="inline-flex items-center px-2.5 py-1 rounded text-[12px] font-medium bg-good-bg text-good border border-good-line max-w-full [overflow-wrap:anywhere] break-words"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[13px] text-muted">
                      No listed skills matched directly.
                    </p>
                  )}
                </div>

                <div className="rounded-panel bg-ground border border-hairline p-3.5 space-y-2.5 min-w-0">
                  <div className="flex items-center justify-between gap-2 text-[13px]">
                    <span className="font-medium text-ink">Missing skills</span>
                    <span className="font-mono text-[12px] px-1.5 py-0.5 rounded bg-pot-bg text-pot border border-pot-line tabular-nums">
                      {match.missing_skills.length}
                    </span>
                  </div>
                  {match.missing_skills.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {match.missing_skills.map((skill, idx) => (
                        <span
                          key={idx}
                          className="inline-flex items-center px-2.5 py-1 rounded text-[12px] font-medium bg-pot-bg text-pot border border-pot-line max-w-full [overflow-wrap:anywhere] break-words"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="text-[13px] text-muted">
                      No missing required skills flagged.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right: 7-Dimension Rule Breakdown + Radar */}
        <div className="xl:col-span-5 rounded-panel bg-ground border border-hairline p-4 space-y-3.5 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h4 className="text-[14px] font-medium text-ink">
                Seven rule weights
              </h4>
              <p className="text-[12px] font-mono text-muted">
                100 pt total · V3.1 breakdown (does not set fit)
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowRadar((prev) => !prev)}
              aria-expanded={showRadar}
              className="inline-flex items-center gap-1.5 min-h-[44px] px-3 rounded-ctrl border border-hairline-strong bg-card text-[13px] font-medium text-ink hover:bg-surface transition-colors shrink-0"
            >
              <span>{showRadar ? 'Hide radar' : 'Show radar'}</span>
              {showRadar ? (
                <ChevronUpIcon className="w-4 h-4" aria-hidden="true" />
              ) : (
                <ChevronDownIcon className="w-4 h-4" aria-hidden="true" />
              )}
            </button>
          </div>

          {showRadar && (
            <SevenAxisRadar scores={match.scores} unscored={unscored} />
          )}

          <div className="space-y-2.5 pt-1">
            {SCORE_DIMENSIONS.map((dim) => {
              const isUnscored = unscored.has(dim);
              const rawScore = match.scores[dim] ?? 0;
              const weight = DIMENSION_WEIGHTS[dim];

              return (
                <div key={dim} className="space-y-1">
                  <div className="flex items-center justify-between gap-2 text-[13px]">
                    <span
                      className={
                        isUnscored ? 'text-muted' : 'text-ink font-medium'
                      }
                    >
                      {DIMENSION_LABELS[dim]}{' '}
                      <span className="font-mono text-[11px] text-muted">
                        ({weight}w)
                      </span>
                    </span>
                    {isUnscored ? (
                      <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-surface text-muted border border-hairline">
                        not scored
                      </span>
                    ) : (
                      <span className="font-mono text-[12px] font-medium text-ink tabular-nums">
                        {Math.round(rawScore)}/100
                      </span>
                    )}
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-surface overflow-hidden">
                    {isUnscored ? (
                      <div className="w-full h-full border-t border-dashed border-hairline-strong" />
                    ) : (
                      <div
                        data-score-bar
                        className="h-full rounded-full bar-brand origin-left"
                        style={{
                          width: `${Math.max(0, Math.min(100, rawScore))}%`,
                        }}
                      />
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer Actions: Literal Quotes, Job Text, and Feedback */}
      <div className="pt-4 border-t border-hairline flex flex-col lg:flex-row lg:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          {evidenceEntries.length > 0 && (
            <button
              type="button"
              onClick={() => setShowEvidence((prev) => !prev)}
              aria-expanded={showEvidence}
              className="inline-flex items-center gap-2 min-h-[44px] px-3.5 rounded-ctrl border border-hairline-strong bg-ground text-[14px] font-medium text-ink hover:bg-surface transition-colors"
            >
              <ChatBubbleBottomCenterTextIcon
                className="w-4 h-4 text-muted shrink-0"
                aria-hidden="true"
              />
              <span>
                {showEvidence
                  ? 'Hide literal quotes'
                  : `Inspect literal quotes (${evidenceEntries.length})`}
              </span>
              {showEvidence ? (
                <ChevronUpIcon
                  className="w-4 h-4 text-muted"
                  aria-hidden="true"
                />
              ) : (
                <ChevronDownIcon
                  className="w-4 h-4 text-muted"
                  aria-hidden="true"
                />
              )}
            </button>
          )}

          {match.job_text && (
            <button
              type="button"
              onClick={() => setShowJobText((prev) => !prev)}
              aria-expanded={showJobText}
              className="inline-flex items-center gap-2 min-h-[44px] px-3.5 rounded-ctrl border border-hairline-strong bg-ground text-[14px] font-medium text-ink hover:bg-surface transition-colors"
            >
              <DocumentTextIcon
                className="w-4 h-4 text-muted shrink-0"
                aria-hidden="true"
              />
              <span>
                {showJobText ? 'Hide job posting' : 'Read job posting'}
              </span>
              {showJobText ? (
                <ChevronUpIcon
                  className="w-4 h-4 text-muted"
                  aria-hidden="true"
                />
              ) : (
                <ChevronDownIcon
                  className="w-4 h-4 text-muted"
                  aria-hidden="true"
                />
              )}
            </button>
          )}
        </div>

        {/* Human Feedback Bar (/api/feedback) */}
        {typeof runId === 'number' && (
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={isSavingFeedback}
              onClick={() => handleVoteClick(1)}
              aria-pressed={vote === 1}
              className={`inline-flex items-center gap-1.5 min-h-[44px] px-3 rounded-ctrl border text-[13px] font-medium transition-colors disabled:opacity-50 ${
                vote === 1
                  ? 'bg-good-bg text-good border-good-line'
                  : 'bg-ground text-ink border-hairline-strong hover:bg-surface'
              }`}
            >
              <HandThumbUpIcon
                className="w-4 h-4 shrink-0"
                aria-hidden="true"
              />
              <span>Good match</span>
            </button>

            <button
              type="button"
              disabled={isSavingFeedback}
              onClick={() => handleVoteClick(-1)}
              aria-pressed={vote === -1}
              className={`inline-flex items-center gap-1.5 min-h-[44px] px-3 rounded-ctrl border text-[13px] font-medium transition-colors disabled:opacity-50 ${
                vote === -1
                  ? 'bg-bad-bg text-bad border-bad-line'
                  : 'bg-ground text-ink border-hairline-strong hover:bg-surface'
              }`}
            >
              <HandThumbDownIcon
                className="w-4 h-4 shrink-0"
                aria-hidden="true"
              />
              <span>Bad match</span>
            </button>

            <div className="inline-flex items-center gap-2">
              <label
                htmlFor={selectId}
                className="text-[13px] text-muted whitespace-nowrap"
              >
                Verdict
              </label>
              <select
                id={selectId}
                value={overrideGrade}
                disabled={isSavingFeedback}
                onChange={handleGradeChange}
                className="min-h-[44px] px-2.5 rounded-ctrl border border-hairline-strong bg-ground text-[14px] text-ink disabled:opacity-50"
              >
                <option value="">Model grade</option>
                <option value="Good Fit">Good Fit</option>
                <option value="Potential Fit">Potential Fit</option>
                <option value="No Fit">No Fit</option>
              </select>
            </div>

            {feedbackSavedFlag && (
              <span
                role="status"
                className="font-mono text-[12px] text-good px-1"
              >
                Saved
              </span>
            )}
          </div>
        )}
      </div>

      {/* Expandable Literal Quotes Drawer */}
      {showEvidence && evidenceEntries.length > 0 && (
        <div className="rounded-panel bg-ground border border-hairline p-4 space-y-3">
          <h5 className="text-[14px] font-medium text-ink">
            Literal quotes from CV and job posting
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {evidenceEntries.map(([dim, quotes]) => (
              <div
                key={dim}
                className="rounded-ctrl bg-card border border-hairline p-3 space-y-1.5 min-w-0"
              >
                <p className="text-[13px] font-mono font-medium text-ink capitalize">
                  {dim}
                </p>
                <ul className="space-y-1 pl-4 list-disc text-[13px] text-body leading-relaxed">
                  {quotes.map((q, qIdx) => (
                    <li
                      key={qIdx}
                      className="[overflow-wrap:anywhere] break-words"
                    >
                      {q}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expandable Job Posting Drawer */}
      {showJobText && match.job_text && (
        <div className="rounded-panel bg-ground border border-hairline p-4 space-y-2">
          <h5 className="text-[14px] font-medium text-ink">
            {isPastedJob
              ? 'Your pasted job description'
              : `Full CareerViet job posting (${match.jd})`}
          </h5>
          <pre className="text-[13px] leading-relaxed text-body font-mono max-h-80 overflow-y-auto whitespace-pre-wrap [overflow-wrap:anywhere] break-words m-0">
            {match.job_text}
          </pre>
        </div>
      )}
    </article>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({
  result,
  onNewAnalysis,
  onFeedbackSaved,
}) => {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [showRemoved, setShowRemoved] = useState(false);
  const [showFormula, setShowFormula] = useState(false);
  const matches = result.matches || [];
  const removedLines = result.injection_removed ?? [];
  const skippedCount = result.skipped ? Object.keys(result.skipped).length : 0;
  const isPastedRun =
    !result.retrieved && matches.some((m) => m.jd === 'pasted');

  useEffect(() => {
    setShowRemoved(false);
  }, [result]);

  useGSAP(
    () => {
      if (!rootRef.current) return;
      const mm = gsap.matchMedia();

      mm.add(
        MOTION_QUERIES,
        (context) => {
          if (!context.conditions?.canAnimate || !rootRef.current) return;

          const snapInt = gsap.utils.snap(1);
          const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

          tl.from('[data-result-card]', {
            y: 16,
            opacity: 0.85,
            duration: 0.46,
            stagger: 0.065,
            clearProps: 'transform,opacity',
          }).from(
            '[data-score-bar]',
            {
              scaleX: 0,
              transformOrigin: 'left center',
              duration: 0.58,
              stagger: 0.02,
              ease: 'power2.out',
              clearProps: 'transform',
            },
            0.16
          );

          const arcs = gsap.utils.toArray<SVGCircleElement>(
            '[data-fit-ring-arc]',
            rootRef.current
          );
          arcs.forEach((arc, i) => {
            const circ = Number(arc.dataset.circumference || 163.36);
            const targetOffset = Number(arc.dataset.targetOffset || 0);
            tl.fromTo(
              arc,
              { strokeDashoffset: circ },
              {
                strokeDashoffset: targetOffset,
                duration: 0.85,
                ease: 'power3.out',
              },
              0.12 + i * 0.075
            );
          });

          const nums = gsap.utils.toArray<HTMLElement>(
            '[data-fit-number]',
            rootRef.current
          );
          nums.forEach((numEl, i) => {
            const target = Number(numEl.dataset.targetFit || 0);
            const state = { val: 0 };
            tl.to(
              state,
              {
                val: target,
                duration: 0.8,
                ease: 'power2.out',
                onUpdate: () => {
                  numEl.textContent = String(snapInt(state.val));
                },
              },
              0.12 + i * 0.075
            );
          });

          requestAnimationFrame(() => {
            ScrollTrigger.refresh();
          });

          return () => {
            arcs.forEach((arc) => {
              if (arc.dataset.targetOffset) {
                arc.style.strokeDashoffset = arc.dataset.targetOffset;
              }
            });
            nums.forEach((numEl) => {
              if (numEl.dataset.targetFit) {
                numEl.textContent = numEl.dataset.targetFit;
              }
            });
          };
        },
        rootRef
      );

      return () => mm.revert();
    },
    {
      scope: rootRef,
      dependencies: [result.run_id, result.as_of, matches.length],
      revertOnUpdate: true,
    }
  );

  return (
    <div ref={rootRef} className="w-full space-y-5">
      {/* Results Summary Bento Card */}
      <div
        data-result-card
        className="bento-card bento-card--highlight p-5 sm:p-6 space-y-4"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1.5 min-w-0">
            <div className="flex flex-wrap items-center gap-2 text-[12px] font-mono text-muted">
              <span className="px-2 py-0.5 rounded bg-accent-soft text-accent-text border border-accent-line">
                {result.ranking_model}
              </span>
              <span>as of {result.as_of}</span>
              {typeof result.run_id === 'number' && (
                <>
                  <span aria-hidden="true">·</span>
                  <span>run #{result.run_id}</span>
                </>
              )}
            </div>
            <h2 className="text-[22px] sm:text-[26px] leading-[1.2] text-ink">
              {isPastedRun
                ? 'Matched against your pasted job'
                : `${matches.length} best of ${result.pool} candidate jobs`}{' '}
              ·{' '}
              <span className="font-mono tabular-nums">
                {result.seconds.toFixed(1)} s
              </span>
            </h2>
            {skippedCount > 0 && (
              <p className="text-[13px] text-pot">
                {skippedCount} candidate posting{skippedCount === 1 ? '' : 's'}{' '}
                skipped during scoring due to incomplete job schema.
              </p>
            )}
          </div>

          {onNewAnalysis && (
            <button
              type="button"
              onClick={onNewAnalysis}
              className="btn-secondary shrink-0 w-full sm:w-auto"
            >
              <ArrowPathIcon className="w-4 h-4" aria-hidden="true" />
              <span>Match another CV</span>
            </button>
          )}
        </div>

        {/* Fit Score Formula Strip */}
        <div className="pt-3.5 border-t border-hairline space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
            <p className="text-[13px] text-muted leading-relaxed">
              <span className="font-mono text-ink">
                fit = 100 / (1 + exp(-logit))
              </span>{' '}
              from <span className="font-mono">{result.ranking_model}</span>{' '}
              (5 rule scores + 3 cosines); verdict cut-offs around 32 and 86.
              The 7-axis rule score is shown beside fit for breakdown only.
            </p>
            <button
              type="button"
              onClick={() => setShowFormula((prev) => !prev)}
              aria-expanded={showFormula}
              className="inline-flex items-center gap-1.5 min-h-[44px] px-3 rounded-ctrl border border-hairline-strong bg-ground text-[13px] font-medium text-ink hover:bg-surface transition-colors shrink-0 self-start sm:self-auto"
            >
              <span>{showFormula ? 'Hide formula' : 'Score formula'}</span>
              {showFormula ? (
                <ChevronUpIcon className="w-4 h-4 text-muted" aria-hidden="true" />
              ) : (
                <ChevronDownIcon className="w-4 h-4 text-muted" aria-hidden="true" />
              )}
            </button>
          </div>

          {showFormula && (
            <div className="rounded-panel bg-ground border border-hairline p-4 space-y-3 text-[13px] text-body">
              <div className="space-y-1">
                <p className="font-mono text-ink tabular-nums">
                  fit = 100 / (1 + exp(-logit)), &nbsp;logit ={' '}
                  {FIT_FORMULA_INTERCEPT} + Σ(weight × feature)
                </p>
                <p className="text-muted leading-relaxed">
                  Verdict cut-offs (<span className="font-mono">grade_cutoffs</span>{' '}
                  <span className="font-mono tabular-nums">-0.754985</span> and{' '}
                  <span className="font-mono tabular-nums">1.828859</span>):{' '}
                  <span className="text-ink">No Fit</span> below{' '}
                  <span className="font-mono tabular-nums">32.0</span>,{' '}
                  <span className="text-ink">Potential Fit</span> from{' '}
                  <span className="font-mono tabular-nums">32.0</span> to{' '}
                  <span className="font-mono tabular-nums">86.1</span>,{' '}
                  <span className="text-ink">Good Fit</span> at{' '}
                  <span className="font-mono tabular-nums">86.2+</span>.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                {FIT_FORMULA_FEATURES.map((feat) => (
                  <div
                    key={feat.name}
                    className="rounded-ctrl bg-card border border-hairline px-3 py-2 space-y-0.5 min-w-0"
                  >
                    <div className="flex items-center justify-between gap-2 font-mono text-[12px] tabular-nums">
                      <span className="text-ink font-medium">{feat.name}</span>
                      <span className="text-muted">{feat.weight}</span>
                    </div>
                    <p className="text-[12px] text-muted leading-snug">
                      {feat.detail}
                    </p>
                  </div>
                ))}
              </div>

              <p className="text-[12px] text-muted leading-relaxed">
                The 7-dimension <span className="font-mono">rule_score</span>{' '}
                (Skills 30, Title 20, Experience 15, Education 10, Industry 10,
                Location 8, Salary 7) is shown beside each match for breakdown,
                but does not set the 0–100 fit score or verdict.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Calm Notice When Injection Guard Removed Lines */}
      {removedLines.length > 0 && (
        <section
          data-result-card
          aria-label="Removed document lines"
          className="rounded-panel bg-surface border border-hairline-strong p-4 sm:p-5 space-y-3"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="space-y-1 min-w-0">
              <p className="text-[14px] font-medium text-ink">
                {formatRemovedSummary(removedLines)}
              </p>
              <p className="text-[13px] text-muted leading-relaxed">
                Lines that look like instructions to the AI are left out before
                extraction so they do not affect the match.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowRemoved((prev) => !prev)}
              aria-expanded={showRemoved}
              className="inline-flex items-center gap-1.5 min-h-[44px] px-3 rounded-ctrl border border-hairline-strong bg-card text-[13px] font-medium text-ink hover:bg-ground transition-colors shrink-0 self-start sm:self-auto"
            >
              <span>
                {showRemoved ? 'Hide removed text' : 'Show removed text'}
              </span>
              {showRemoved ? (
                <ChevronUpIcon className="w-4 h-4 text-muted" aria-hidden="true" />
              ) : (
                <ChevronDownIcon className="w-4 h-4 text-muted" aria-hidden="true" />
              )}
            </button>
          </div>

          {showRemoved && (
            <ul className="space-y-2 pt-1">
              {removedLines.map((item, idx) => (
                <li
                  key={`${item.source}-${item.line}-${idx}`}
                  className="rounded-ctrl bg-card border border-hairline p-3 space-y-1.5"
                >
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] font-mono text-muted">
                    <span className="text-ink font-medium">
                      {item.source === 'jd' ? 'Job description' : 'CV'}
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="tabular-nums">line {item.line}</span>
                    <span aria-hidden="true">·</span>
                    <span>
                      {INJECTION_KIND_LABELS[item.kind] ??
                        item.kind.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <pre className="text-[13px] leading-relaxed text-body font-mono whitespace-pre-wrap [overflow-wrap:anywhere] break-words m-0">
                    {item.text}
                  </pre>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {/* Ranked Job Cards */}
      {matches.length === 0 ? (
        <div
          data-result-card
          className="bento-card p-8 text-center space-y-2"
        >
          <p className="text-[17px] font-medium text-ink">
            No matching jobs were returned for this CV.
          </p>
          <p className="text-[15px] text-muted">
            Upload a fuller CV or increase the number of returned jobs.
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          {matches.map((match, index) => (
            <JobMatchCard
              key={`${result.run_id ?? result.as_of}-${match.jd}-${index}`}
              match={match}
              rank={index + 1}
              runId={result.run_id}
              initialFeedback={result.feedback?.[match.jd]}
              onFeedbackSaved={onFeedbackSaved}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;
