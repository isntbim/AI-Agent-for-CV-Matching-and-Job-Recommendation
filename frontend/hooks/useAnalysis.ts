import { useState, useCallback, useEffect, useRef } from 'react';
import { matchCv, MatchCvParams, MatchResponse } from '../services/api';
import toast from 'react-hot-toast';

export interface UseAnalysisReturn {
  isLoading: boolean;
  elapsedSeconds: number;
  results: MatchResponse | null;
  error: string | null;
  runMatch: (params: MatchCvParams) => Promise<void>;
  reset: () => void;
}

export const useAnalysis = (): UseAnalysisReturn => {
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [results, setResults] = useState<MatchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stopTimer();
    };
  }, [stopTimer]);

  const runMatch = useCallback(
    async ({ file, text, top = 5, asOf }: MatchCvParams) => {
      const trimmedText = text?.trim() || '';
      if (!file && !trimmedText) {
        const msg = 'Upload a CV file or paste your CV text to start matching.';
        setError(msg);
        toast.error(msg);
        return;
      }

      if (!file && trimmedText.length < 200) {
        const msg =
          'CV text is too short (fewer than 200 characters). If your PDF is scanned, export it as text first.';
        setError(msg);
        toast.error(msg);
        return;
      }

      setIsLoading(true);
      setElapsedSeconds(0);
      setError(null);
      setResults(null);

      stopTimer();
      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
      }, 500);

      try {
        const response = await matchCv({
          file,
          text: file ? undefined : trimmedText,
          top,
          asOf,
        });
        setResults(response);
        toast.success(
          `Found ${response.matches.length} matching jobs in ${response.seconds.toFixed(1)}s`
        );
      } catch (err) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : 'Failed to match CV against job pool.';
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        stopTimer();
        setIsLoading(false);
      }
    },
    [stopTimer]
  );

  const reset = useCallback(() => {
    stopTimer();
    setIsLoading(false);
    setElapsedSeconds(0);
    setResults(null);
    setError(null);
  }, [stopTimer]);

  return {
    isLoading,
    elapsedSeconds,
    results,
    error,
    runMatch,
    reset,
  };
};

export default useAnalysis;
