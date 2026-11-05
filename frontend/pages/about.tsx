import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Header from '../components/Header';
import Footer from '../components/Footer';
import {
  CpuChipIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';

const AboutPage: React.FC = () => {
  const pipelineStages = [
    {
      stage: 'Stage 1',
      title: 'Fine-Tuned Qwen2.5-7B (QLoRA) Extraction',
      icon: CpuChipIcon,
      description:
        'Uploaded CVs (.pdf, .docx, .txt) or pasted resume text are parsed into clean text and passed to our QLoRA fine-tuned Qwen2.5-7B extractor. The model outputs structured V3.1 JSON covering skills, normalized job titles, work experience, education, industry domains, location, and evidence spans.',
    },
    {
      stage: 'Stage 2',
      title: 'LangGraph Validate-Repair-Retry Loop',
      icon: ArrowPathIcon,
      description:
        'Extracted JSON is validated against strict Pydantic schemas and business rules. If a field has a structural or schema issue, a deterministic repair pass and bounded retry loop resolve errors before downstream scoring.',
    },
    {
      stage: 'Stage 3',
      title: 'BGE-M3 + BM25 Hybrid Candidate Retrieval',
      icon: MagnifyingGlassIcon,
      description:
        'Instead of scoring every job sequentially, hybrid retrieval combines multilingual BGE-M3 dense embeddings with BM25 lexical search to select the top 20 candidate jobs out of 361 indexed CareerViet postings.',
    },
    {
      stage: 'Stage 4',
      title: 'V3.1 7-Dimension Rule + hybrid_tfidf_v1 Scorer',
      icon: ChartBarIcon,
      description:
        'Each candidate job is evaluated across 7 dimensions (skills, title, experience, education, industry, location, salary). The hybrid_tfidf_v1 ranking model combines the V3.1 rule score with semantic embedding and TF-IDF similarity into a calibrated 0–100 fit score and fit grade.',
    },
  ];

  const dimensions = [
    {
      name: 'Skills',
      detail:
        'Matches extracted CV skills against required and optional JD skills using exact and taxonomy-aware skill relations.',
    },
    {
      name: 'Title',
      detail:
        'Compares normalized candidate job roles and seniority level against the target job role.',
    },
    {
      name: 'Experience',
      detail:
        'Evaluates total and role-relevant years of experience relative to the minimum requirement in the job posting.',
    },
    {
      name: 'Education',
      detail:
        'Compares degree level and field of study when the job posting specifies an educational requirement.',
    },
    {
      name: 'Industry',
      detail:
        'Checks overlap between the candidate’s background domains and the job’s industry domains.',
    },
    {
      name: 'Location',
      detail:
        'Checks city/country alignment or remote work compatibility when both documents state a location.',
    },
    {
      name: 'Salary',
      detail:
        'Checks compensation compatibility when both the CV and job posting specify salary expectations or ranges.',
    },
  ];

  const techStack = [
    {
      name: 'Qwen2.5-7B (QLoRA)',
      category: 'Structured CV & JD Extraction',
    },
    {
      name: 'LangGraph + Pydantic',
      category: 'Validate-Repair-Retry State Machine',
    },
    {
      name: 'BGE-M3 + BM25',
      category: 'Hybrid Candidate Retrieval (Top 20 of 361)',
    },
    {
      name: 'hybrid_tfidf_v1 + V3.1 Scorer',
      category: '7-Dimension Rule + Hybrid Ranking',
    },
    {
      name: 'FastAPI + Python',
      category: 'Backend Matching Service',
    },
    {
      name: 'Next.js 14 + TypeScript + Tailwind',
      category: 'Frontend Web Interface',
    },
  ];

  return (
    <>
      <Head>
        <title>About the Pipeline - ResuMatch CareerViet Matcher</title>
        <meta
          name="description"
          content="Architecture of our ATS CV-JD matching system: Qwen2.5-7B QLoRA extractor, LangGraph validate-repair-retry, BGE-M3 + BM25 retrieval, and V3.1 7-dimension hybrid scoring."
        />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50 flex flex-col">
        <Header />

        <main className="flex-1 pt-24 pb-16">
          {/* Hero Section */}
          <section className="py-12">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium mb-6">
                  <CpuChipIcon className="w-4 h-4" />
                  <span>Capstone System Architecture</span>
                </div>

                <h1 className="text-4xl md:text-5xl font-bold text-secondary-900 mb-6">
                  Explainable CV-to-Jobs Matching with{' '}
                  <span className="gradient-text">Qwen2.5-7B & Hybrid Search</span>
                </h1>

                <p className="text-lg text-secondary-600 leading-relaxed max-w-3xl mx-auto">
                  This project matches a candidate’s CV against 361 real
                  CareerViet job postings. Every match decision is backed by
                  schema-validated extraction, hybrid dense + lexical retrieval,
                  and a 7-dimension scorer that separates supported evidence from
                  missing information.
                </p>
              </motion.div>
            </div>
          </section>

          {/* 4-Stage Pipeline Architecture */}
          <section id="pipeline" className="py-12 bg-white">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-secondary-900 mb-3">
                  End-to-End Matching Pipeline
                </h2>
                <p className="text-base text-secondary-600 max-w-2xl mx-auto">
                  How a single CV upload is transformed into ranked job
                  recommendations.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {pipelineStages.map((item, idx) => (
                  <motion.div
                    key={item.stage}
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: idx * 0.08 }}
                    viewport={{ once: true }}
                    className="bg-secondary-50/60 rounded-2xl p-6 border border-secondary-200/80 space-y-3"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-bold text-primary-700 bg-primary-100 px-2.5 py-1 rounded-full">
                        {item.stage}
                      </span>
                      <item.icon className="w-6 h-6 text-primary-600" />
                    </div>
                    <h3 className="text-xl font-bold text-secondary-900">
                      {item.title}
                    </h3>
                    <p className="text-sm text-secondary-600 leading-relaxed">
                      {item.description}
                    </p>
                  </motion.div>
                ))}
              </div>
            </div>
          </section>

          {/* 7 Scoring Dimensions */}
          <section id="scoring" className="py-16">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold text-secondary-900 mb-3">
                  The 7 Scoring Dimensions
                </h2>
                <p className="text-base text-secondary-600 max-w-2xl mx-auto">
                  When a CV or job posting does not contain enough evidence for
                  an axis (such as salary or location), the system marks that
                  axis as <span className="font-mono font-semibold">not scored</span>{' '}
                  rather than guessing.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {dimensions.map((dim) => (
                  <div
                    key={dim.name}
                    className="bg-white rounded-xl p-5 border border-secondary-200 shadow-soft space-y-2"
                  >
                    <div className="flex items-center space-x-2">
                      <CheckCircleIcon className="w-5 h-5 text-primary-600 shrink-0" />
                      <h3 className="font-bold text-secondary-900">
                        {dim.name}
                      </h3>
                    </div>
                    <p className="text-sm text-secondary-600 leading-relaxed">
                      {dim.detail}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Technology Stack */}
          <section className="py-12 bg-white">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="text-center mb-10">
                <h2 className="text-2xl md:text-3xl font-bold text-secondary-900 mb-2">
                  Technology Stack
                </h2>
                <p className="text-sm text-secondary-600">
                  Core components powering extraction, retrieval, and ranking.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                {techStack.map((tech) => (
                  <div
                    key={tech.name}
                    className="p-5 rounded-xl bg-secondary-50 border border-secondary-200 flex items-start gap-3"
                  >
                    <ChatBubbleBottomCenterTextIcon className="w-5 h-5 text-primary-600 shrink-0 mt-0.5" />
                    <div>
                      <div className="font-bold text-secondary-900 text-sm">
                        {tech.name}
                      </div>
                      <div className="text-xs text-secondary-600 mt-0.5">
                        {tech.category}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-12 text-center">
                <Link
                  href="/analyze"
                  className="btn-primary inline-flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <span>Try the Matcher with Your CV</span>
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

export default AboutPage;
