import axios from 'axios';

// '' = same origin: the Docker image's Next.js server forwards /api/* to the API container (next.config.js)
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes: first-time CV extraction with Qwen2.5-7B + repair/retry on 12 GB GPU
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

export type PipelineStep = 'extract' | 'retrieve' | 'score' | 'rank';

export const SCORE_DIMENSIONS: ScoreDimension[] = [
  'skills',
  'title',
  'experience',
  'education',
  'industry',
  'location',
  'salary',
];

export const DIMENSION_WEIGHTS: Record<ScoreDimension, number> = {
  skills: 30,
  title: 20,
  experience: 15,
  education: 10,
  industry: 10,
  location: 8,
  salary: 7,
};

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
  /** O5 explainer's paragraph (checked against the scores), or the fixed template when it failed the check. */
  summary?: string;
  summary_source?: 'o5' | 'template';
  evidence: Record<string, string[]>;
  job_text: string;
}

export interface SavedFeedback {
  vote: 1 | -1;
  grade: FitGrade | null;
  comment: string | null;
}

export type InjectionSource = 'cv' | 'jd';

export interface InjectionFinding {
  line: number;
  kind: string;
  text: string;
  source: InjectionSource;
}

export interface MatchResponse {
  run_id?: number;
  as_of: string;
  retrieved: boolean;
  pool: number;
  seconds: number;
  skipped: Record<string, string>;
  ranking_model: string;
  matches: JobMatch[];
  feedback?: Record<string, SavedFeedback>;
  injection_removed?: InjectionFinding[];
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
  jdText?: string;
}

export interface HistoryTopMatch {
  jd: string;
  title?: string | null;
  grade?: FitGrade | null;
  fit?: number | null;
}

export interface HistoryRunSummary {
  run_id: number;
  created: string;
  seconds: number;
  feedback: number;
  top: HistoryTopMatch[];
}

export interface FeedbackPayload {
  run_id: number;
  jd: string;
  vote: 1 | -1;
  grade?: FitGrade | null;
  comment?: string | null;
}

export interface StreamCallbacks {
  onToken?: (data: { tokens: number; tail: string }) => void;
  onStep?: (data: { step: PipelineStep; seconds: number }) => void;
  signal?: AbortSignal;
}

const NOT_SCORED_PREFIX =
  'Not scored (evidence missing or inconclusive, neutral value used):';

export function todayIsoDate(): string {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 10);
}

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

function buildMatchFormData({
  file,
  text,
  top = 5,
  asOf,
  jdText,
}: MatchCvParams): FormData {
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
  const trimmedJd =
    typeof jdText === 'string'
      ? jdText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim()
      : '';
  if (trimmedJd.length > 0) {
    formData.append('jd_text', trimmedJd);
  }
  return formData;
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
    const response = await api.get<HealthResponse>('/api/health', {
      timeout: 5000,
    });
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, 'Failed to check matching service health.')
    );
  }
}

export async function matchCv(params: MatchCvParams): Promise<MatchResponse> {
  const formData = buildMatchFormData(params);
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

export async function matchCvStream(
  params: MatchCvParams,
  callbacks: StreamCallbacks = {}
): Promise<MatchResponse> {
  const formData = buildMatchFormData(params);
  const response = await fetch(`${API_BASE_URL}/api/match/stream`, {
    method: 'POST',
    body: formData,
    signal: callbacks.signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail =
      typeof body?.detail === 'string'
        ? body.detail
        : Array.isArray(body?.detail)
          ? body.detail
              .map((d: { msg?: string } | string) =>
                typeof d === 'string' ? d : d?.msg || JSON.stringify(d)
              )
              .join('; ')
          : `The server returned status ${response.status}.`;
    throw new Error(detail);
  }

  if (!response.body) {
    return matchCv(params);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: MatchResponse | null = null;
  let done = false;

  while (!done) {
    const chunk = await reader.read();
    done = chunk.done;
    buffer += decoder.decode(chunk.value || new Uint8Array(), {
      stream: !done,
    });

    let boundary = buffer.indexOf('\n\n');
    while (boundary >= 0) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf('\n\n');

      const lines = rawEvent.split('\n');
      const kind = lines
        .find((line) => line.startsWith('event: '))
        ?.slice(7)
        .trim();
      const dataStr = lines
        .find((line) => line.startsWith('data: '))
        ?.slice(6);

      if (!kind || !dataStr) continue;

      const payload = JSON.parse(dataStr);
      if (kind === 'token') {
        callbacks.onToken?.(payload);
      } else if (kind === 'step') {
        callbacks.onStep?.(payload);
      } else if (kind === 'result') {
        finalResult = payload as MatchResponse;
      } else if (kind === 'error') {
        throw new Error(
          payload?.detail || 'The matching pipeline reported an error.'
        );
      }
    }
  }

  if (!finalResult) {
    throw new Error(
      'The stream closed before the ranked result arrived. Check Recent runs if extraction completed.'
    );
  }

  return finalResult;
}

export async function getHistory(limit = 8): Promise<HistoryRunSummary[]> {
  try {
    const response = await api.get<HistoryRunSummary[]>('/api/history', {
      params: { limit },
      timeout: 10000,
    });
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, 'Failed to load recent matching runs.')
    );
  }
}

export async function getHistoryRun(runId: number): Promise<MatchResponse> {
  try {
    const response = await api.get<MatchResponse>(`/api/history/${runId}`, {
      timeout: 15000,
    });
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, `Failed to load run #${runId}.`)
    );
  }
}

export async function submitFeedback(
  payload: FeedbackPayload
): Promise<{ saved: boolean }> {
  try {
    const response = await api.post<{ saved: boolean }>(
      '/api/feedback',
      payload,
      { timeout: 10000 }
    );
    return response.data;
  } catch (error) {
    throw new Error(
      extractErrorMessage(error, 'Failed to save match feedback.')
    );
  }
}

export default api;
