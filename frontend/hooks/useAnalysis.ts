import { useState, useCallback, useEffect, useRef } from 'react';
import {
  getHistory,
  getHistoryRun,
  HistoryRunSummary,
  MatchCvParams,
  matchCvStream,
  MatchResponse,
  PipelineStep,
} from '../services/api';
import toast from 'react-hot-toast';

const STEP_ORDER: PipelineStep[] = ['extract', 'retrieve', 'score', 'rank'];

export interface UseAnalysisReturn {
  isLoading: boolean;
  isLoadingHistoryRun: boolean;
  elapsedSeconds: number;
  activeStep: PipelineStep | null;
  completedSteps: Partial<Record<PipelineStep, number>>;
  streamTokens: number;
  streamTail: string;
  results: MatchResponse | null;
  history: HistoryRunSummary[];
  error: string | null;
  runMatch: (params: MatchCvParams) => Promise<boolean>;
  loadRun: (runId: number) => Promise<boolean>;
  refreshHistory: () => Promise<void>;
  reset: () => void;
}

export const useAnalysis = (): UseAnalysisReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistoryRun, setIsLoadingHistoryRun] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [activeStep, setActiveStep] = useState<PipelineStep | null>(null);
  const [completedSteps, setCompletedSteps] = useState<
    Partial<Record<PipelineStep, number>>
  >({});
  const [streamTokens, setStreamTokens] = useState(0);
  const [streamTail, setStreamTail] = useState('');
  const [results, setResults] = useState<MatchResponse | null>(null);
  const [history, setHistory] = useState<HistoryRunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const refreshHistory = useCallback(async () => {
    try {
      const runs = await getHistory(8);
      setHistory(runs);
    } catch {
      // History is a convenience panel; matching still works if history cannot be fetched.
    }
  }, []);

  useEffect(() => {
    refreshHistory();
    return () => {
      stopTimer();
      abortRef.current?.abort();
    };
  }, [refreshHistory, stopTimer]);

  const runMatch = useCallback(
    async ({ file, text, top = 5, asOf }: MatchCvParams): Promise<boolean> => {
      const trimmedText = text?.trim() || '';
      if (!file && !trimmedText) {
        const msg =
          'Select a CV file (.pdf, .docx, or .txt) or paste at least 200 characters of CV text.';
        setError(msg);
        toast.error(msg);
        return false;
      }

      if (!file && trimmedText.length < 200) {
        const msg =
          'Paste at least 200 characters of CV text so the extractor can score your history.';
        setError(msg);
        toast.error(msg);
        return false;
      }

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setIsLoading(true);
      setElapsedSeconds(0);
      setActiveStep('extract');
      setCompletedSteps({});
      setStreamTokens(0);
      setStreamTail('');
      setError(null);
      setResults(null);

      stopTimer();
      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
      }, 400);

      try {
        const response = await matchCvStream(
          {
            file,
            text: file ? undefined : trimmedText,
            top,
            asOf,
          },
          {
            signal: controller.signal,
            onToken: ({ tokens, tail }) => {
              setStreamTokens(tokens);
              setStreamTail(tail);
            },
            onStep: ({ step, seconds }) => {
              setCompletedSteps((prev) => ({ ...prev, [step]: seconds }));
              setStreamTail('');
              const idx = STEP_ORDER.indexOf(step);
              const nextStep =
                idx >= 0 && idx + 1 < STEP_ORDER.length
                  ? STEP_ORDER[idx + 1]
                  : null;
              setActiveStep(nextStep);
            },
          }
        );

        stopTimer();
        setActiveStep(null);
        setIsLoading(false);
        setResults(response);
        void refreshHistory();
        toast.success(
          `Ranked ${response.matches.length} jobs in ${response.seconds.toFixed(1)} s`
        );
        return true;
      } catch (err) {
        if (controller.signal.aborted) {
          return false;
        }
        const errorMessage =
          err instanceof Error
            ? err.message
            : 'Failed to match CV against the CareerViet corpus.';
        setError(errorMessage);
        toast.error(errorMessage);
        return false;
      } finally {
        stopTimer();
        setIsLoading(false);
        setActiveStep(null);
      }
    },
    [refreshHistory, stopTimer]
  );

  const loadRun = useCallback(
    async (runId: number): Promise<boolean> => {
      if (isLoading) return false;
      setIsLoadingHistoryRun(true);
      setError(null);
      try {
        const saved = await getHistoryRun(runId);
        setResults(saved);
        return true;
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : `Could not load run #${runId}.`;
        setError(msg);
        toast.error(msg);
        return false;
      } finally {
        setIsLoadingHistoryRun(false);
      }
    },
    [isLoading]
  );

  const reset = useCallback(() => {
    abortRef.current?.abort();
    stopTimer();
    setIsLoading(false);
    setElapsedSeconds(0);
    setActiveStep(null);
    setCompletedSteps({});
    setStreamTokens(0);
    setStreamTail('');
    setResults(null);
    setError(null);
  }, [stopTimer]);

  return {
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
  };
};

export default useAnalysis;
