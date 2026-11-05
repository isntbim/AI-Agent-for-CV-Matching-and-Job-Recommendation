import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  DocumentTextIcon,
  SparklesIcon,
  ArrowPathIcon,
  ChatBubbleBottomCenterTextIcon,
} from '@heroicons/react/24/outline';
import {
  FitGrade,
  JobMatch,
  MatchResponse,
  SCORE_DIMENSIONS,
  ScoreDimension,
  parseUnscoredDimensions,
} from '../services/api';

interface AnalysisResultsProps {
  result: MatchResponse;
  onNewAnalysis?: () => void;
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

function getGradeMeta(grade: FitGrade) {
  switch (grade) {
    case 'Good Fit':
      return {
        badgeClass: 'bg-success-50 text-success-800 border-success-200',
        ringColor: '#16a34a',
        Icon: CheckCircleIcon,
      };
    case 'Potential Fit':
      return {
        badgeClass: 'bg-warning-50 text-warning-800 border-warning-200',
        ringColor: '#d97706',
        Icon: ExclamationTriangleIcon,
      };
    case 'No Fit':
    default:
      return {
        badgeClass: 'bg-error-50 text-error-800 border-error-200',
        ringColor: '#dc2626',
        Icon: XCircleIcon,
      };
  }
}

interface FitScoreRingProps {
  fit: number;
  grade: FitGrade;
}

const FitScoreRing: React.FC<FitScoreRingProps> = ({ fit, grade }) => {
  const clamped = Math.max(0, Math.min(100, fit));
  const radius = 28;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;
  const { ringColor } = getGradeMeta(grade);

  return (
    <div
      className="relative w-18 h-18 flex items-center justify-center shrink-0"
      aria-label={`Overall fit score ${clamped.toFixed(1)} out of 100`}
    >
      <svg className="w-16 h-16 -rotate-90" viewBox="0 0 68 68">
        <circle
          cx="34"
          cy="34"
          r={radius}
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="6"
        />
        <circle
          cx="34"
          cy="34"
          r={radius}
          fill="none"
          stroke={ringColor}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-sm font-bold text-secondary-900 leading-none">
          {Math.round(clamped)}
        </span>
        <span className="text-[10px] text-secondary-500 leading-none mt-0.5">
          /100
        </span>
      </div>
    </div>
  );
};

interface RadarChartProps {
  scores: Record<ScoreDimension, number>;
  unscored: Set<ScoreDimension>;
}

const SevenAxisRadar: React.FC<RadarChartProps> = ({ scores, unscored }) => {
  const cx = 140;
  const cy = 125;
  const maxR = 72;
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

  // For the polygon shape, scored dimensions use their actual score/100;
  // unscored dimensions sit at 0.35 on the dashed spoke so the polygon stays visually balanced
  // while clearly marked as greyed/unscored dots and "not scored" in the legend.
  const polygonPoints = SCORE_DIMENSIONS.map((dim, i) => {
    const isUnscored = unscored.has(dim);
    const ratio = isUnscored
      ? 0.35
      : Math.max(0.05, Math.min(1, (scores[dim] ?? 0) / 100));
    const pt = pointAt(i, ratio);
    return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
  }).join(' ');

  return (
    <div className="flex flex-col lg:flex-row items-center gap-6 bg-secondary-50/70 rounded-xl p-4 border border-secondary-200/80">
      <div className="w-full max-w-[280px] shrink-0">
        <svg
          viewBox="0 0 280 250"
          className="w-full h-auto overflow-visible"
          role="img"
          aria-label="7-dimension score radar chart"
        >
          {/* Reference heptagons */}
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
                stroke="#e2e8f0"
                strokeWidth="1"
              />
            );
          })}

          {/* Spokes */}
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
                stroke={isUnscored ? '#cbd5e1' : '#94a3b8'}
                strokeWidth={isUnscored ? '1.25' : '1.5'}
                strokeDasharray={isUnscored ? '4 4' : undefined}
              />
            );
          })}

          {/* Radar polygon */}
          <polygon
            points={polygonPoints}
            fill="rgba(37, 99, 235, 0.14)"
            stroke="#2563eb"
            strokeWidth="2"
          />

          {/* Axis dots & labels */}
          {SCORE_DIMENSIONS.map((dim, i) => {
            const isUnscored = unscored.has(dim);
            const ratio = isUnscored
              ? 0.35
              : Math.max(0.05, Math.min(1, (scores[dim] ?? 0) / 100));
            const dot = pointAt(i, ratio);
            const labelPt = pointAt(i, 1.24);
            const a = angleForIndex(i);
            const cos = Math.cos(a);
            const textAnchor =
              cos > 0.25 ? 'start' : cos < -0.25 ? 'end' : 'middle';

            return (
              <g key={dim}>
                <circle
                  cx={dot.x}
                  cy={dot.y}
                  r={isUnscored ? 4 : 4.5}
                  fill={isUnscored ? '#f1f5f9' : '#2563eb'}
                  stroke={isUnscored ? '#94a3b8' : '#ffffff'}
                  strokeWidth="1.75"
                />
                <text
                  x={labelPt.x}
                  y={labelPt.y}
                  textAnchor={textAnchor}
                  dominantBaseline="middle"
                  className={`text-[11px] ${
                    isUnscored
                      ? 'fill-secondary-400 italic'
                      : 'fill-secondary-800 font-semibold'
                  }`}
                >
                  {DIMENSION_LABELS[dim]}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* 7-Dimension Legend */}
      <div className="w-full flex-1 space-y-2.5">
        <div className="flex items-center justify-between text-xs text-secondary-500 pb-1 border-b border-secondary-200">
          <span className="font-semibold text-secondary-700">
            7-Dimension Rule Breakdown
          </span>
          <span>Scored vs. unscored axes</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
          {SCORE_DIMENSIONS.map((dim) => {
            const isUnscored = unscored.has(dim);
            const rawScore = scores[dim] ?? 0;

            return (
              <div key={dim} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span
                    className={
                      isUnscored
                        ? 'text-secondary-400 font-normal'
                        : 'text-secondary-800 font-medium'
                    }
                  >
                    {DIMENSION_LABELS[dim]}
                  </span>
                  {isUnscored ? (
                    <span className="px-1.5 py-0.5 rounded bg-secondary-200/70 text-secondary-600 text-[11px] font-medium">
                      not scored
                    </span>
                  ) : (
                    <span className="font-semibold text-secondary-900">
                      {Math.round(rawScore)}/100
                    </span>
                  )}
                </div>
                <div className="w-full h-1.5 rounded-full bg-secondary-200 overflow-hidden">
                  {isUnscored ? (
                    <div className="w-full h-full border-t border-dashed border-secondary-400" />
                  ) : (
                    <div
                      className="h-full rounded-full bg-primary-600 transition-all duration-300"
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
  );
};

interface JobMatchCardProps {
  match: JobMatch;
  rank: number;
}

const JobMatchCard: React.FC<JobMatchCardProps> = ({ match, rank }) => {
  const [showRadar, setShowRadar] = useState(rank === 1);
  const [showJobText, setShowJobText] = useState(false);
  const [showEvidence, setShowEvidence] = useState(false);

  const unscored = parseUnscoredDimensions(match.explanation || []);
  const { badgeClass, Icon: GradeIcon } = getGradeMeta(match.grade);

  // Skip line 0 which repeats the headline
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

  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: Math.min(rank * 0.05, 0.3) }}
      className="bg-white rounded-2xl border border-secondary-200 shadow-soft overflow-hidden"
    >
      {/* Card Header */}
      <div className="p-5 sm:p-6 border-b border-secondary-100">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3.5 min-w-0">
            <span className="inline-flex items-center justify-center px-2.5 py-1 rounded-lg bg-primary-50 border border-primary-200 text-primary-700 font-bold text-sm shrink-0">
              #{rank}
            </span>
            <div className="min-w-0">
              <h3 className="text-lg sm:text-xl font-bold text-secondary-900 break-words">
                {match.title || 'Untitled Position'}
              </h3>
              <div className="flex flex-wrap items-center gap-2 mt-1 text-xs text-secondary-500">
                <span className="font-mono bg-secondary-100 text-secondary-700 px-2 py-0.5 rounded">
                  {match.jd}
                </span>
                <span>•</span>
                <span>Retrieval rank #{match.retrieval_rank}</span>
                <span>•</span>
                <span>Rule score {match.rule_score.toFixed(1)}/100</span>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between sm:justify-end gap-4 shrink-0">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${badgeClass}`}
            >
              <GradeIcon className="w-4 h-4 shrink-0" />
              <span>{match.grade}</span>
            </span>
            <FitScoreRing fit={match.fit} grade={match.grade} />
          </div>
        </div>
      </div>

      {/* Card Body */}
      <div className="p-5 sm:p-6 space-y-5">
        {/* Plain-English Explanation Bullets */}
        {explanationBullets.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-secondary-500">
              Why this job matched
            </h4>
            <ul className="space-y-1.5 text-sm text-secondary-700">
              {explanationBullets.map((line, idx) => {
                const isNotScoredLine = line.startsWith('Not scored');
                return (
                  <li
                    key={idx}
                    className={`flex items-start gap-2 ${
                      isNotScoredLine ? 'text-xs text-secondary-500 italic' : ''
                    }`}
                  >
                    <span
                      className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                        isNotScoredLine ? 'bg-secondary-400' : 'bg-primary-600'
                      }`}
                    />
                    <span className="break-words">{line}</span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {/* Skill Chips */}
        <div className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-secondary-500">
            Skill Alignment
          </h4>
          {!hasAnySkills ? (
            <p className="text-sm text-secondary-500 italic">
              No skill overlap recorded
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3.5 rounded-xl bg-success-50/50 border border-success-200/70">
                <div className="text-xs font-semibold text-success-800 mb-2 flex items-center justify-between">
                  <span>Has (Matched Skills)</span>
                  <span className="px-1.5 py-0.5 rounded bg-success-100 text-success-900">
                    {match.matched_skills.length}
                  </span>
                </div>
                {match.matched_skills.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {match.matched_skills.map((skill, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-success-100 text-success-900 border border-success-200"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-secondary-500">
                    None of the listed job skills were directly matched.
                  </p>
                )}
              </div>

              <div className="p-3.5 rounded-xl bg-warning-50/50 border border-warning-200/70">
                <div className="text-xs font-semibold text-warning-900 mb-2 flex items-center justify-between">
                  <span>Missing Skills</span>
                  <span className="px-1.5 py-0.5 rounded bg-warning-100 text-warning-900">
                    {match.missing_skills.length}
                  </span>
                </div>
                {match.missing_skills.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {match.missing_skills.map((skill, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-warning-100/80 text-warning-900 border border-warning-200"
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-secondary-500">
                    No missing required skills flagged.
                  </p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* 7-Dimension Radar Section (Expanded on #1 by default, toggleable on all cards) */}
        <div>
          <button
            type="button"
            onClick={() => setShowRadar((prev) => !prev)}
            aria-expanded={showRadar}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-primary-700 hover:text-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 rounded px-2 py-1 -ml-2"
          >
            {showRadar ? (
              <ChevronUpIcon className="w-4 h-4" />
            ) : (
              <ChevronDownIcon className="w-4 h-4" />
            )}
            <span>
              {showRadar
                ? 'Hide 7-dimension radar chart'
                : 'Show 7-dimension radar chart'}
            </span>
          </button>

          {showRadar && (
            <div className="mt-2">
              <SevenAxisRadar scores={match.scores} unscored={unscored} />
            </div>
          )}
        </div>

        {/* Action toggles: Job Posting & Evidence */}
        <div className="pt-2 border-t border-secondary-100 flex flex-wrap items-center gap-3">
          {match.job_text && (
            <button
              type="button"
              onClick={() => setShowJobText((prev) => !prev)}
              aria-expanded={showJobText}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-secondary-100 text-secondary-800 hover:bg-secondary-200 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <DocumentTextIcon className="w-4 h-4 text-secondary-600" />
              <span>
                {showJobText ? 'Hide job posting' : 'Read the job posting'}
              </span>
              {showJobText ? (
                <ChevronUpIcon className="w-3.5 h-3.5" />
              ) : (
                <ChevronDownIcon className="w-3.5 h-3.5" />
              )}
            </button>
          )}

          {evidenceEntries.length > 0 && (
            <button
              type="button"
              onClick={() => setShowEvidence((prev) => !prev)}
              aria-expanded={showEvidence}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-secondary-100 text-secondary-800 hover:bg-secondary-200 transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <ChatBubbleBottomCenterTextIcon className="w-4 h-4 text-secondary-600" />
              <span>
                {showEvidence
                  ? 'Hide extracted evidence'
                  : `Evidence (${evidenceEntries.length})`}
              </span>
              {showEvidence ? (
                <ChevronUpIcon className="w-3.5 h-3.5" />
              ) : (
                <ChevronDownIcon className="w-3.5 h-3.5" />
              )}
            </button>
          )}
        </div>

        {/* Expandable Job Text */}
        {showJobText && match.job_text && (
          <div className="rounded-xl bg-secondary-900 text-secondary-100 p-4 text-xs leading-relaxed max-h-80 overflow-y-auto whitespace-pre-wrap font-mono">
            {match.job_text}
          </div>
        )}

        {/* Expandable Evidence */}
        {showEvidence && evidenceEntries.length > 0 && (
          <div className="rounded-xl bg-secondary-50 border border-secondary-200 p-4 space-y-3">
            <h5 className="text-xs font-semibold uppercase tracking-wider text-secondary-600">
              Supporting Quotes from CV & Job Posting
            </h5>
            <div className="space-y-2.5">
              {evidenceEntries.map(([dim, quotes]) => (
                <div key={dim} className="text-xs">
                  <span className="font-semibold text-secondary-800 capitalize">
                    {dim}:
                  </span>
                  <ul className="mt-1 space-y-1 pl-4 list-disc text-secondary-700">
                    {quotes.map((q, qIdx) => (
                      <li key={qIdx} className="break-words">
                        {q}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.article>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({
  result,
  onNewAnalysis,
}) => {
  const matches = result.matches || [];

  return (
    <div className="w-full max-w-5xl mx-auto space-y-6">
      {/* Results Summary Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-2xl border border-secondary-200 shadow-soft p-5 sm:p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
      >
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 text-xs font-semibold text-primary-700 bg-primary-50 border border-primary-200 px-2.5 py-1 rounded-full">
            <SparklesIcon className="w-3.5 h-3.5" />
            <span>Ranked by {result.ranking_model}</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-secondary-900">
            {matches.length} best of {result.pool} candidate jobs ·{' '}
            {result.seconds.toFixed(1)} s
          </h2>
          <p className="text-xs sm:text-sm text-secondary-600">
            Candidate pool retrieved via hybrid BGE-M3 + BM25 search and scored
            as of <span className="font-mono">{result.as_of}</span>.
          </p>
        </div>

        {onNewAnalysis && (
          <button
            type="button"
            onClick={onNewAnalysis}
            className="btn-secondary inline-flex items-center justify-center gap-2 text-sm shrink-0 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <ArrowPathIcon className="w-4 h-4" />
            <span>Match Another CV</span>
          </button>
        )}
      </motion.div>

      {/* Ranked Job Cards */}
      {matches.length === 0 ? (
        <div className="bg-white rounded-2xl border border-secondary-200 p-8 text-center space-y-3">
          <p className="text-base font-medium text-secondary-800">
            No matching jobs were returned for this CV.
          </p>
          <p className="text-sm text-secondary-500">
            Try uploading a more detailed CV or increasing the number of
            returned jobs.
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          {matches.map((match, index) => (
            <JobMatchCard
              key={`${match.jd}-${index}`}
              match={match}
              rank={index + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default AnalysisResults;
