import React, { useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion, AnimatePresence } from 'framer-motion';
import Header from '../components/Header';
import Footer from '../components/Footer';
import {
  QuestionMarkCircleIcon,
  ChevronDownIcon,
  DocumentTextIcon,
  ClockIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';

const HelpPage: React.FC = () => {
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  const faqs = [
    {
      question: 'What file formats and sizes are supported?',
      answer:
        'You can upload text-based PDF (.pdf), Microsoft Word (.docx), or plain text (.txt) files up to 5 MB. Alternatively, you can switch to the "Paste CV Text" tab on the Match CV page and paste at least 200 characters of resume text directly.',
    },
    {
      question:
        'Why was my PDF rejected with "fewer than 200 characters"?',
      answer:
        'If a PDF was created by scanning a paper document or exporting pages as images without an embedded text layer, the parser extracts fewer than 200 characters. Export your CV as a text-based PDF, DOCX, or TXT file—or copy and paste your resume text into the "Paste CV Text" tab.',
    },
    {
      question:
        'Why does a new CV take about 60–120 seconds on the first run, while repeat runs are much faster?',
      answer:
        'On the first upload of a new CV, our fine-tuned Qwen2.5-7B (QLoRA) model extracts structured V3.1 resume facts and runs the LangGraph validate-repair-retry loop on GPU, which takes roughly 60–120 seconds. Once extracted, the structured CV is cached on disk keyed by the SHA-256 hash of its text, so matching the same CV again (or changing the number of returned jobs) completes in a few seconds.',
    },
    {
      question: 'What do the 7 scoring dimensions mean?',
      answer:
        'Every candidate job is evaluated on 7 axes: (1) Skills — overlap with required and optional JD skills including taxonomy relations; (2) Title — role family and seniority alignment; (3) Experience — years of relevant work experience; (4) Education — degree level and field of study; (5) Industry — domain overlap; (6) Location — city/country compatibility; and (7) Salary — compensation alignment.',
    },
    {
      question:
        'What does "Not scored" mean on the 7-dimension radar chart and explanation?',
      answer:
        'When either the CV or the job posting does not provide enough evidence to evaluate a dimension (for example, when neither document mentions salary or when location is unspecified), the backend substitutes a neutral internal value so the overall rule score is not unfairly penalized, and lists those axes under "Not scored (evidence missing or inconclusive, neutral value used)". On the radar chart and breakdown legend, those axes are drawn with dashed spokes and labelled "not scored" rather than shown as real scores.',
    },
    {
      question:
        'How are the top jobs selected from the 361 CareerViet job postings?',
      answer:
        'First, hybrid retrieval combining BGE-M3 dense embeddings and BM25 lexical search narrows the 361 CareerViet jobs down to the 20 closest candidate postings. Second, the hybrid_tfidf_v1 ranking model combines the 7-dimension V3.1 rule score with semantic and TF-IDF similarity to rank those 20 jobs and return your top 3, 5, or 10 matches.',
    },
    {
      question:
        'What does "Extractor offline — cached CVs only" in the header mean?',
      answer:
        'The header checks GET /api/health. When the local Ollama server hosting our fine-tuned Qwen2.5-7B extractor is running, the badge shows "361 jobs ready". If Ollama is not running, the badge shows "Extractor offline — cached CVs only": previously extracted CVs that are already in the SHA-256 cache will still match, while brand-new CVs require starting `ollama serve`.',
    },
  ];

  const quickTips = [
    {
      icon: DocumentTextIcon,
      title: 'Use Selectable Text',
      description:
        'Upload a text-based PDF, DOCX, or TXT file (up to 5 MB) with at least 200 characters of text.',
    },
    {
      icon: ClockIcon,
      title: 'SHA-256 Extraction Cache',
      description:
        'First-time extraction takes ~60–120 s on GPU. Repeat queries for the same CV text use the cached extraction.',
    },
    {
      icon: ChartBarIcon,
      title: 'Inspect All 7 Axes',
      description:
        'Expand the 7-dimension radar chart, matched/missing skill chips, and supporting quotes on each job card.',
    },
    {
      icon: ExclamationTriangleIcon,
      title: 'Check Service Status',
      description:
        'Verify the status badge in the top navigation bar to ensure both the FastAPI server and Qwen extractor are online.',
    },
  ];

  return (
    <>
      <Head>
        <title>Help & FAQ - ResuMatch CareerViet Matcher</title>
        <meta
          name="description"
          content="Frequently asked questions about uploading CVs, Qwen2.5-7B extraction latency and SHA-256 caching, 7-dimension scoring, and unscored dimensions."
        />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex flex-col">
        <Header />

        <main className="flex-1 pt-24 pb-16">
          {/* Hero Section */}
          <section className="py-10">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium mb-5">
                  <QuestionMarkCircleIcon className="w-4 h-4" />
                  <span>Usage Guide & Troubleshooting</span>
                </div>

                <h1 className="text-4xl md:text-5xl font-bold text-secondary-900 mb-4">
                  Help & Frequently Asked Questions
                </h1>

                <p className="text-lg text-secondary-600 max-w-2xl mx-auto">
                  Everything you need to know about uploading your CV, how the
                  Qwen2.5-7B extractor and SHA-256 cache work, and how to read
                  your 7-dimension match results.
                </p>
              </motion.div>
            </div>
          </section>

          {/* Quick Tips */}
          <section className="py-8">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {quickTips.map((tip) => (
                  <div
                    key={tip.title}
                    className="bg-white rounded-2xl p-6 border border-secondary-200 shadow-soft space-y-2.5"
                  >
                    <div className="w-10 h-10 rounded-xl bg-primary-50 border border-primary-200 flex items-center justify-center text-primary-600">
                      <tip.icon className="w-5 h-5" />
                    </div>
                    <h2 className="text-base font-bold text-secondary-900">
                      {tip.title}
                    </h2>
                    <p className="text-xs sm:text-sm text-secondary-600 leading-relaxed">
                      {tip.description}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* FAQ Accordion */}
          <section id="faq" className="py-12">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="space-y-4">
                {faqs.map((faq, index) => {
                  const isOpen = openFaq === index;
                  return (
                    <div
                      key={faq.question}
                      className="bg-white rounded-xl border border-secondary-200 shadow-soft overflow-hidden"
                    >
                      <button
                        type="button"
                        onClick={() => setOpenFaq(isOpen ? null : index)}
                        aria-expanded={isOpen}
                        className="w-full px-6 py-4 text-left flex items-center justify-between gap-4 hover:bg-secondary-50 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
                      >
                        <span className="font-semibold text-secondary-900 text-sm sm:text-base">
                          {faq.question}
                        </span>
                        <ChevronDownIcon
                          className={`w-5 h-5 text-secondary-500 shrink-0 transition-transform duration-200 ${
                            isOpen ? 'rotate-180' : ''
                          }`}
                        />
                      </button>

                      <AnimatePresence initial={false}>
                        {isOpen && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.25 }}
                            className="overflow-hidden"
                          >
                            <div className="px-6 pb-5 pt-1 text-sm text-secondary-600 leading-relaxed border-t border-secondary-100">
                              {faq.answer}
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>

              <div className="mt-10 text-center">
                <Link
                  href="/analyze"
                  className="btn-primary inline-flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <span>Go to Match CV</span>
                  <ArrowRightIcon className="w-4 h-4" />
                </Link>
              </div>
            </div>
          </section>
        </main>

        <Footer />
      </div>
    </>
  );
};

export default HelpPage;
