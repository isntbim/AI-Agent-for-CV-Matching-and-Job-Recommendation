import React, { useState } from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import Header from '../components/Header';
import Footer from '../components/Footer';
import FileUpload from '../components/FileUpload';
import AnalysisResults from '../components/AnalysisResults';
import { useAnalysis } from '../hooks/useAnalysis';
import {
  SparklesIcon,
  DocumentTextIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  CpuChipIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline';

type InputTab = 'file' | 'text';
const TOP_OPTIONS = [3, 5, 10] as const;

const AnalyzePage: React.FC = () => {
  const [inputTab, setInputTab] = useState<InputTab>('file');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [cvText, setCvText] = useState<string>('');
  const [topCount, setTopCount] = useState<number>(5);

  const { isLoading, elapsedSeconds, results, error, runMatch, reset } =
    useAnalysis();

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (inputTab === 'file') {
      await runMatch({ file: selectedFile, top: topCount });
    } else {
      await runMatch({ text: cvText, top: topCount });
    }
  };

  const handleNewAnalysis = () => {
    setSelectedFile(null);
    setCvText('');
    reset();
  };

  const trimmedTextLength = cvText.trim().length;
  const canSubmit =
    !isLoading &&
    ((inputTab === 'file' && selectedFile !== null) ||
      (inputTab === 'text' && trimmedTextLength >= 200));

  return (
    <>
      <Head>
        <title>Match Your CV - ResuMatch CareerViet Matcher</title>
        <meta
          name="description"
          content="Upload your CV or paste your resume text to rank the best-fitting CareerViet jobs with our fine-tuned Qwen2.5-7B extractor and hybrid retrieval pipeline."
        />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex flex-col">
        <Header />

        <main className="flex-1 pt-24 pb-16">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Page Header */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="text-center mb-10"
            >
              <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-1.5 rounded-full text-xs sm:text-sm font-medium mb-4">
                <SparklesIcon className="w-4 h-4" />
                <span>361 CareerViet Jobs · Hybrid BGE-M3 + BM25 Retrieval</span>
              </div>

              <h1 className="text-3xl md:text-4xl font-bold text-secondary-900 mb-3">
                Find Your Best-Matching Jobs
              </h1>

              <p className="text-base sm:text-lg text-secondary-600 max-w-2xl mx-auto">
                Upload your CV once. Our pipeline extracts your skills and
                experience, retrieves the 20 closest job postings, and scores
                each job across 7 dimensions.
              </p>
            </motion.div>

            {/* Error Banner */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                role="alert"
                className="mb-8 rounded-xl bg-error-50 border border-error-200 p-4 sm:p-5 flex items-start gap-3.5"
              >
                <ExclamationTriangleIcon className="w-6 h-6 text-error-600 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <h2 className="text-sm font-semibold text-error-900">
                    Could not complete CV matching
                  </h2>
                  <p className="text-sm text-error-800 mt-1 break-words">
                    {error}
                  </p>
                </div>
              </motion.div>
            )}

            {/* Loading State */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-white rounded-2xl border border-secondary-200 shadow-soft p-8 sm:p-10 text-center space-y-6 mb-8"
              >
                <div className="relative w-16 h-16 mx-auto flex items-center justify-center rounded-2xl bg-primary-50 border border-primary-200 text-primary-600">
                  <CpuChipIcon className="w-8 h-8 animate-pulse" />
                </div>

                <div className="space-y-2 max-w-xl mx-auto">
                  <h2 className="text-lg sm:text-xl font-bold text-secondary-900">
                    Reading your CV with our fine-tuned model (about a minute
                    for a new CV)
                  </h2>
                  <p className="text-sm text-secondary-600">
                    Extracting structured V3.1 schema facts with Qwen2.5-7B
                    (validate-repair-retry), retrieving top 20 candidate jobs,
                    and computing 7-dimension fit scores.
                  </p>
                </div>

                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary-100 text-secondary-800 text-sm font-mono font-semibold">
                  <ClockIcon className="w-4 h-4 text-primary-600" />
                  <span>Elapsed: {elapsedSeconds}s</span>
                </div>

                <div className="max-w-md mx-auto grid grid-cols-3 gap-2 text-xs text-secondary-600 pt-2">
                  <div className="p-2.5 rounded-lg bg-secondary-50 border border-secondary-200">
                    1. Qwen2.5-7B CV Extraction
                  </div>
                  <div className="p-2.5 rounded-lg bg-secondary-50 border border-secondary-200">
                    2. BGE-M3 + BM25 Top-20 Pool
                  </div>
                  <div className="p-2.5 rounded-lg bg-secondary-50 border border-secondary-200">
                    3. 7-Dimension Fit Ranking
                  </div>
                </div>
              </motion.div>
            )}

            {/* Input Form or Results */}
            {!results ? (
              !isLoading && (
                <motion.form
                  onSubmit={handleSubmit}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.05 }}
                  className="bg-white rounded-2xl border border-secondary-200 shadow-soft p-6 sm:p-8 space-y-6"
                >
                  {/* Input Mode Tabs */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-secondary-100">
                    <div
                      role="tablist"
                      aria-label="CV input method"
                      className="inline-flex rounded-xl bg-secondary-100 p-1 self-start"
                    >
                      <button
                        type="button"
                        role="tab"
                        aria-selected={inputTab === 'file'}
                        onClick={() => setInputTab('file')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                          inputTab === 'file'
                            ? 'bg-white text-secondary-900 shadow-sm'
                            : 'text-secondary-600 hover:text-secondary-900'
                        }`}
                      >
                        Upload File (.pdf, .docx, .txt)
                      </button>
                      <button
                        type="button"
                        role="tab"
                        aria-selected={inputTab === 'text'}
                        onClick={() => setInputTab('text')}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                          inputTab === 'text'
                            ? 'bg-white text-secondary-900 shadow-sm'
                            : 'text-secondary-600 hover:text-secondary-900'
                        }`}
                      >
                        Paste CV Text
                      </button>
                    </div>

                    {/* Top Jobs Count Selector */}
                    <div className="flex items-center gap-2.5">
                      <label
                        htmlFor="top-jobs-select"
                        className="text-xs sm:text-sm font-medium text-secondary-700"
                      >
                        Jobs to return:
                      </label>
                      <select
                        id="top-jobs-select"
                        value={topCount}
                        onChange={(e) => setTopCount(Number(e.target.value))}
                        className="rounded-lg border border-secondary-300 bg-white px-3 py-1.5 text-sm font-medium text-secondary-900 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      >
                        {TOP_OPTIONS.map((n) => (
                          <option key={n} value={n}>
                            Top {n} jobs
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Tab Content */}
                  {inputTab === 'file' ? (
                    <div className="space-y-2">
                      <label className="block text-sm font-semibold text-secondary-900">
                        CV File Upload
                      </label>
                      <FileUpload
                        onFileSelect={handleFileSelect}
                        selectedFile={selectedFile}
                        disabled={isLoading}
                      />
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="cv-text-input"
                          className="block text-sm font-semibold text-secondary-900"
                        >
                          Paste Your CV Text
                        </label>
                        <span
                          className={`text-xs font-mono ${
                            trimmedTextLength >= 200
                              ? 'text-success-700'
                              : 'text-secondary-500'
                          }`}
                        >
                          {trimmedTextLength} chars (minimum 200)
                        </span>
                      </div>
                      <textarea
                        id="cv-text-input"
                        rows={10}
                        value={cvText}
                        onChange={(e) => setCvText(e.target.value)}
                        placeholder="Paste the full text of your CV here (summary, skills, work experience, education — at least 200 characters)..."
                        className="input-field resize-y font-mono text-sm"
                      />
                      <p className="text-xs text-secondary-500 flex items-center gap-1.5">
                        <DocumentTextIcon className="w-4 h-4 text-primary-600 shrink-0" />
                        <span>
                          Include your job titles, skills, work history, and
                          education for accurate 7-dimension scoring.
                        </span>
                      </p>
                    </div>
                  )}

                  {/* Submit Button */}
                  <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4">
                    <p className="text-xs text-secondary-500">
                      Identical CVs are cached by SHA-256 hash so repeat lookups
                      return in a few seconds.
                    </p>
                    <button
                      type="submit"
                      disabled={!canSubmit}
                      className={`w-full sm:w-auto px-7 py-3.5 rounded-xl font-semibold text-base flex items-center justify-center gap-2.5 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
                        canSubmit
                          ? 'bg-gradient-to-r from-primary-600 to-accent-600 hover:from-primary-700 hover:to-accent-700 text-white shadow-medium'
                          : 'bg-secondary-200 text-secondary-500 cursor-not-allowed'
                      }`}
                    >
                      <MagnifyingGlassIcon className="w-5 h-5" />
                      <span>Find My Best-Matching Jobs</span>
                    </button>
                  </div>
                </motion.form>
              )
            ) : (
              <div className="space-y-6">
                <AnalysisResults
                  result={results}
                  onNewAnalysis={handleNewAnalysis}
                />

                <div className="flex justify-center pt-4">
                  <button
                    type="button"
                    onClick={handleNewAnalysis}
                    className="btn-secondary inline-flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    <ArrowPathIcon className="w-4 h-4" />
                    <span>Start Another CV Match</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>

        <Footer />
      </div>
    </>
  );
};

export default AnalyzePage;
