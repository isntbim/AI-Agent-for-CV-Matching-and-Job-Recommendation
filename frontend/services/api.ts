import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes: first-time CV extraction with Qwen + repair/retry can take ~120 s on GPU
});

export type FitGrade = 'Good Fit' | 'Potential Fit' | 'No Fit';

export type ScoreDimension =
  | 'skills'
  | 'title'
  | 'experience'
  | 'education'
  | 'industry'
  | 'location'
  | 'salary';

export const SCORE_DIMENSIONS: ScoreDimension[] = [
  'skills',
  'title',
  'experience',
  'education',
  'industry',
  'location',
  'salary',
];

export interface DimensionScores {
  skills: number;
  title: number;
  experience: number;
  education: number;
  industry: number;
  location: number;
  salary: number;
}

export interface JobMatch {
  jd: string;
  title: string;
  retrieval_rank: number;
  fit: number;
  grade: FitGrade;
  rule_score: number;
  scores: DimensionScores;
  matched_skills: string[];
  missing_skills: string[];
  explanation: string[];
  evidence: Record<string, string[]>;
  job_text: string;
}

export interface MatchResponse {
  as_of: string;
  retrieved: boolean;
  pool: number;
  seconds: number;
  skipped: Record<string, string>;
  ranking_model: string;
  matches: JobMatch[];
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | string;
  extractor: string;
  jobs: number;
  ranking_model: string;
}

export interface MatchCvParams {
  file?: File | null;
  text?: string;
  top?: number;
  asOf?: string;
}

const NOT_SCORED_PREFIX =
  'Not scored (evidence missing or inconclusive, neutral value used):';

/**
 * Extracts dimension names that the backend marked as unscored in the explanation lines.
 */
export function parseUnscoredDimensions(
  explanation: string[]
): Set<ScoreDimension> {
  const unscored = new Set<ScoreDimension>();
  for (const line of explanation) {
    if (line.startsWith(NOT_SCORED_PREFIX)) {
      const rest = line
        .slice(NOT_SCORED_PREFIX.length)
        .replace(/\.$/, '')
        .trim();
      for (const part of rest.split(',')) {
        const dim = part.trim().toLowerCase() as ScoreDimension;
        if (SCORE_DIMENSIONS.includes(dim)) {
          unscored.add(dim);
        }
      }
    }
  }
  return unscored;
}

function extractErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string' && detail.trim().length > 0) {
      return detail;
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) =>
          typeof item === 'string' ? item : item?.msg || JSON.stringify(item)
        )
        .join('; ');
    }
    if (error.code === 'ECONNABORTED') {
      return 'Request timed out after 300 seconds while waiting for CV extraction and job matching.';
    }
    if (typeof error.message === 'string' && error.message.trim().length > 0) {
      return error.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await api.get<HealthResponse>('/api/health');
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, 'Failed to check matching service health.')
    );
  }
}

export async function matchCv({
  file,
  text,
  top = 5,
  asOf,
}: MatchCvParams): Promise<MatchResponse> {
  const formData = new FormData();
  if (file) {
    formData.append('file', file);
  } else if (typeof text === 'string' && text.trim().length > 0) {
    formData.append('text', text);
  }
  formData.append('top', String(top));
  if (asOf) {
    formData.append('as_of', asOf);
  }

  try {
    const response = await api.post<MatchResponse>('/api/match', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, 'Failed to match CV against job pool.')
    );
  }
}

export default api;
