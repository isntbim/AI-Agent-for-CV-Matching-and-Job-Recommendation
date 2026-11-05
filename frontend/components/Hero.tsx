import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  SparklesIcon,
  DocumentTextIcon,
  ChartBarIcon,
  CpuChipIcon,
  ArrowRightIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

const Hero: React.FC = () => {
  const highlights = [
    'Fine-tuned Qwen2.5-7B (QLoRA) CV extraction',
    'Hybrid BGE-M3 + BM25 search across 361 CareerViet jobs',
    '7-dimension V3.1 rule + hybrid_qwen_v1 ranking',
    'Skill-by-skill explanations & supporting quotes',
  ];

  const floatingCards = [
    {
      icon: CpuChipIcon,
      title: 'Qwen2.5-7B Extractor',
      description: 'QLoRA + validate-repair-retry',
      color: 'from-blue-500 to-cyan-500',
      delay: 0.2,
    },
    {
      icon: DocumentTextIcon,
      title: 'Top 20 of 361 Jobs',
      description: 'BGE-M3 + BM25 retrieval',
      color: 'from-purple-500 to-pink-500',
      delay: 0.35,
    },
    {
      icon: ChartBarIcon,
      title: '7-Axis Fit Breakdown',
      description: 'Skills, role, domain & evidence',
      color: 'from-green-500 to-emerald-500',
      delay: 0.5,
    },
  ];

  const previewDimensions = [
    // real values from one run (cv_021 -> jd_042, output/service/e2e_hybrid_cv021.json), not invented numbers
    { name: 'Skills', value: '8/100', width: '8%', scored: true },
    { name: 'Industry', value: '100/100', width: '100%', scored: true },
    { name: 'Experience', value: 'not scored', width: '0%', scored: false },
    { name: 'Location', value: 'not scored', width: '0%', scored: false },
  ];

  return (
    <section className="relative pt-24 pb-16 overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-accent-50" />
      <div className="absolute top-1/4 left-1/4 w-72 h-72 bg-primary-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse-slow" />
      <div
        className="absolute top-1/3 right-1/4 w-72 h-72 bg-accent-200 rounded-full mix-blend-multiply filter blur-xl opacity-30 animate-pulse-slow"
        style={{ animationDelay: '2s' }}
      />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Content */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center lg:text-left"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-primary-100 to-accent-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium mb-6"
            >
              <SparklesIcon className="w-4 h-4" />
              <span>Explainable CV-to-Jobs Matching Pipeline</span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-4xl md:text-5xl lg:text-6xl font-bold text-secondary-900 mb-6 leading-tight"
            >
              Match Your CV Against{' '}
              <span className="gradient-text">361 CareerViet Jobs</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="text-lg md:text-xl text-secondary-600 mb-8 max-w-2xl mx-auto lg:mx-0"
            >
              Upload your CV once. Our fine-tuned Qwen2.5-7B extractor reads
              your skills and experience, retrieves the 20 closest postings via
              hybrid BGE-M3 + BM25 search, and ranks your top matches with a
              7-dimension breakdown.
            </motion.p>

            {/* Feature List */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-8"
            >
              {highlights.map((feature) => (
                <div
                  key={feature}
                  className="flex items-start space-x-2 text-left"
                >
                  <CheckCircleIcon className="w-5 h-5 text-success-600 flex-shrink-0 mt-0.5" />
                  <span className="text-secondary-700 text-sm font-medium">
                    {feature}
                  </span>
                </div>
              ))}
            </motion.div>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
              className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
            >
              <Link
                href="/analyze"
                className="btn-primary inline-flex items-center justify-center space-x-2 text-base px-8 py-4 group focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <span>Find My Best-Matching Jobs</span>
                <ArrowRightIcon className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-200" />
              </Link>

              <Link
                href="/about"
                className="btn-secondary inline-flex items-center justify-center space-x-2 text-base px-8 py-4 focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <span>How the Pipeline Works</span>
              </Link>
            </motion.div>

            {/* Real Pipeline Numbers */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              className="mt-12 pt-8 border-t border-secondary-200 grid grid-cols-3 gap-6"
            >
              <div className="text-center lg:text-left">
                <div className="text-2xl md:text-3xl font-bold text-primary-600">
                  361
                </div>
                <div className="text-xs sm:text-sm text-secondary-600">
                  Indexed CareerViet Jobs
                </div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl md:text-3xl font-bold text-primary-600">
                  20
                </div>
                <div className="text-xs sm:text-sm text-secondary-600">
                  Hybrid Candidate Pool
                </div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl md:text-3xl font-bold text-primary-600">
                  7
                </div>
                <div className="text-xs sm:text-sm text-secondary-600">
                  Scoring Dimensions
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Visual Example Card */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="relative"
          >
            <div className="relative bg-white rounded-3xl shadow-large p-6 sm:p-8 border border-secondary-100">
              <div className="space-y-5">
                {/* Preview Card Header */}
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-11 h-11 bg-gradient-to-br from-primary-600 to-accent-600 rounded-xl flex items-center justify-center text-white font-bold text-sm">
                      #1
                    </div>
                    <div>
                      <h2 className="font-bold text-secondary-900 text-base">
                        Data Analyst & Data Scientist
                      </h2>
                      <p className="text-xs text-secondary-500 font-mono">
                        jd_042 · Retrieval rank #2
                      </p>
                    </div>
                  </div>
                  <span className="px-3 py-1 rounded-full text-xs font-semibold bg-warning-50 text-warning-800 border border-warning-200">
                    Potential Fit · 52/100
                  </span>
                </div>

                {/* Dimension Preview Bars */}
                <div className="space-y-3 pt-1">
                  {previewDimensions.map((item) => (
                    <div key={item.name} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="font-medium text-secondary-700">
                          {item.name}
                        </span>
                        <span
                          className={
                            item.scored
                              ? 'font-semibold text-secondary-900'
                              : 'text-secondary-500 italic'
                          }
                        >
                          {item.value}
                        </span>
                      </div>
                      <div className="w-full bg-secondary-200 rounded-full h-2">
                        {item.scored ? (
                          <div
                            className="bg-gradient-to-r from-primary-600 to-accent-600 h-2 rounded-full"
                            style={{ width: item.width }}
                          />
                        ) : (
                          <div className="w-full h-full border-t border-dashed border-secondary-400" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Matched / Missing Skill Chips Preview */}
                <div className="pt-2 space-y-2 border-t border-secondary-100">
                  <div className="text-xs font-semibold text-secondary-600">
                    Skill Overlap & Evidence
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-success-100 text-success-900 border border-success-200">
                      Has: sql
                    </span>
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-success-100 text-success-900 border border-success-200">
                      Has: python
                    </span>
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-warning-100 text-warning-900 border border-warning-200">
                      Missing: data analysis
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Floating Cards (hidden on narrow mobile to avoid horizontal overflow) */}
            {floatingCards.map((card, index) => (
              <motion.div
                key={card.title}
                initial={{ opacity: 0, y: 16, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, delay: card.delay }}
                className={`hidden xl:block absolute bg-white rounded-2xl shadow-medium p-3.5 border border-secondary-100 ${
                  index === 0
                    ? '-top-6 -left-6'
                    : index === 1
                      ? 'top-1/2 -right-8 transform -translate-y-1/2'
                      : '-bottom-6 left-6'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-9 h-9 bg-gradient-to-br ${card.color} rounded-lg flex items-center justify-center`}
                  >
                    <card.icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-secondary-900 text-xs">
                      {card.title}
                    </h3>
                    <p className="text-[11px] text-secondary-500">
                      {card.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
