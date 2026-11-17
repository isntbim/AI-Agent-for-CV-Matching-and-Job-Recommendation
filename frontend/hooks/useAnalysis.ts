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
  isPastedMode: boolean;
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
  const [isPastedMode, setIsPastedMode] = useState(false);
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
  const reqSeqRef = useRef(0);

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
    async ({
      file,
      text,
      top = 5,
      asOf,
      jdText,
    }: MatchCvParams): Promise<boolean> => {
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

      const trimmedJd =
        jdText?.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim() || '';
      const jdCodePoints = Array.from(trimmedJd).length;
      if (
        jdCodePoints > 0 &&
        (jdCodePoints < 200 || jdCodePoints > 12000)
      ) {
        const msg =
          'Paste between 200 and 12,000 characters for the job description, or leave it blank to search all 361 jobs.';
        setError(msg);
        toast.error(msg);
        return false;
      }
      const pastedMode = jdCodePoints > 0;
      const stepSequence: PipelineStep[] = pastedMode
        ? ['extract', 'score', 'rank']
        : STEP_ORDER;

      const reqId = ++reqSeqRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setIsLoading(true);
      setIsLoadingHistoryRun(false);
      setIsPastedMode(pastedMode);
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
            top: pastedMode ? 1 : top,
            asOf,
            jdText: pastedMode ? trimmedJd : undefined,
          },
          {
            signal: controller.signal,
            onToken: ({ tokens, tail }) => {
              if (reqSeqRef.current !== reqId) return;
              setStreamTokens(tokens);
              setStreamTail(tail);
            },
            onStep: ({ step, seconds }) => {
              if (reqSeqRef.current !== reqId) return;
              setCompletedSteps((prev) => ({ ...prev, [step]: seconds }));
              setStreamTail('');
              const idx = stepSequence.indexOf(step);
              const nextStep =
                idx >= 0 && idx + 1 < stepSequence.length
                  ? stepSequence[idx + 1]
                  : null;
              setActiveStep(nextStep);
            },
          }
        );

        if (reqSeqRef.current !== reqId) {
          return false;
        }
        stopTimer();
        setActiveStep(null);
        setIsLoading(false);
        setResults(response);
        void refreshHistory();
        toast.success(
          pastedMode || !response.retrieved
            ? `Scored your job description in ${response.seconds.toFixed(1)} s`
            : `Ranked ${response.matches.length} jobs in ${response.seconds.toFixed(1)} s`
        );
        return true;
      } catch (err) {
        if (controller.signal.aborted || reqSeqRef.current !== reqId) {
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
        if (reqSeqRef.current === reqId) {
          stopTimer();
          setIsLoading(false);
          setActiveStep(null);
        }
      }
    },
    [refreshHistory, stopTimer]
  );

  const loadRun = useCallback(
    async (runId: number): Promise<boolean> => {
      if (isLoading) return false;
      const reqId = ++reqSeqRef.current;
      setIsLoadingHistoryRun(true);
      setError(null);
      try {
        const saved = await getHistoryRun(runId);
        if (reqSeqRef.current !== reqId) return false;
        setIsPastedMode(!saved.retrieved);
        setResults(saved);
        return true;
      } catch (err) {
        if (reqSeqRef.current !== reqId) return false;
        const msg =
          err instanceof Error ? err.message : `Could not load run #${runId}.`;
        setError(msg);
        toast.error(msg);
        return false;
      } finally {
        if (reqSeqRef.current === reqId) {
          setIsLoadingHistoryRun(false);
        }
      }
    },
    [isLoading]
  );

  const reset = useCallback(() => {
    ++reqSeqRef.current;
    abortRef.current?.abort();
    stopTimer();
    setIsLoading(false);
    setIsLoadingHistoryRun(false);
    setIsPastedMode(false);
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
    isPastedMode,
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
