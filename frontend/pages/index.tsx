import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { motion } from 'framer-motion';
import Header from '../components/Header';
import Hero from '../components/Hero';
import Features from '../components/Features';
import Footer from '../components/Footer';
import {
  CpuChipIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  ChatBubbleBottomCenterTextIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';

const HomePage: React.FC = () => {
  const steps = [
    {
      step: '01',
      title: 'Qwen2.5-7B Extraction',
      description:
        'Fine-tuned Qwen2.5-7B (QLoRA) extracts structured skills, roles, experience, education, and domains from your CV, with a LangGraph validate-repair-retry loop.',
      icon: CpuChipIcon,
    },
    {
      step: '02',
      title: 'Hybrid Candidate Retrieval',
      description:
        'Hybrid search combining BGE-M3 dense embeddings and BM25 lexical matching picks the 20 closest jobs out of 361 CareerViet postings.',
      icon: MagnifyingGlassIcon,
    },
    {
      step: '03',
      title: 'Hybrid Rule + TF-IDF Ranking',
      description:
        'The hybrid_tfidf_v1 ranking model combines 7-dimension V3.1 rule scores with semantic embedding and TF-IDF similarity.',
      icon: ChartBarIcon,
    },
    {
      step: '04',
      title: 'Explainable Results',
      description:
        'Plain-English explanations show matched skills, missing skills, 7-axis score breakdowns, and supporting quotes from your CV and the job.',
      icon: ChatBubbleBottomCenterTextIcon,
    },
  ];

  return (
    <>
      <Head>
        <title>ResuMatch - CareerViet CV-to-Jobs Matching Pipeline</title>
        <meta
          name="description"
          content="Upload your CV once to match against 361 CareerViet job postings using a fine-tuned Qwen2.5-7B extractor, hybrid BGE-M3 + BM25 search, and 7-dimension explainable ranking."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-secondary-50">
        <Header />

        <main>
          <Hero />
          <Features />

          {/* How It Works Section */}
          <section className="py-20 bg-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
                className="text-center mb-16"
              >
                <h2 className="text-3xl md:text-4xl font-bold text-secondary-900 mb-4">
                  How the Pipeline Works
                </h2>
                <p className="text-lg text-secondary-600 max-w-2xl mx-auto">
                  Four transparent stages take your raw CV from document parsing
                  to ranked, explainable job matches.
                </p>
              </motion.div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                {steps.map((item, index) => (
                  <motion.div
                    key={item.step}
                    initial={{ opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    viewport={{ once: true }}
                    className="relative bg-secondary-50/60 rounded-2xl p-6 border border-secondary-200/80"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="w-12 h-12 bg-gradient-to-br from-primary-600 to-accent-600 rounded-xl flex items-center justify-center shadow-sm">
                        <item.icon className="w-6 h-6 text-white" />
                      </div>
                      <span className="text-xs font-mono font-bold text-primary-700 bg-primary-100 px-2.5 py-1 rounded-full">
                        Step {item.step}
                      </span>
                    </div>
                    <h3 className="text-lg font-bold text-secondary-900 mb-2">
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

          {/* CTA Section */}
          <section className="py-20 bg-gradient-to-r from-primary-600 to-accent-600">
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                viewport={{ once: true }}
              >
                <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                  Ready to Find Your Best-Fitting CareerViet Jobs?
                </h2>
                <p className="text-lg text-primary-100 mb-8 max-w-2xl mx-auto">
                  Upload a PDF, DOCX, or TXT resume—or paste your CV text
                  directly—and inspect your ranked matches across all 7 scoring
                  dimensions.
                </p>
                <Link
                  href="/analyze"
                  className="inline-flex items-center gap-2 bg-white text-primary-700 hover:bg-primary-50 font-semibold py-4 px-8 rounded-xl shadow-large hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-white"
                >
                  <span>Match My CV Now</span>
                  <ArrowRightIcon className="w-5 h-5" />
                </Link>
              </motion.div>
            </div>
          </section>
        </main>

        <Footer />
      </div>
    </>
  );
};

export default HomePage;
