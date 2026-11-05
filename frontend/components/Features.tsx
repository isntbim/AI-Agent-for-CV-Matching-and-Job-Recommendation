import React from 'react';
import { motion } from 'framer-motion';
import {
  CpuChipIcon,
  MagnifyingGlassIcon,
  ChartBarIcon,
  DocumentTextIcon,
  ChatBubbleBottomCenterTextIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

const Features: React.FC = () => {
  const features = [
    {
      icon: CpuChipIcon,
      title: 'Fine-Tuned Qwen2.5-7B Extractor',
      description:
        'Extracts structured V3.1 resume facts — skills, normalized roles, work experience, education, domains, and location — using a QLoRA-tuned Qwen2.5-7B model.',
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-50',
      textColor: 'text-blue-600',
    },
    {
      icon: ArrowPathIcon,
      title: 'Validate-Repair-Retry Loop',
      description:
        'A LangGraph state machine validates extracted JSON against strict Pydantic schemas, automatically repairing formatting issues or retrying when schema checks fail.',
      color: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-50',
      textColor: 'text-purple-600',
    },
    {
      icon: MagnifyingGlassIcon,
      title: 'Hybrid BGE-M3 + BM25 Retrieval',
      description:
        'Combines multilingual BGE-M3 dense vector embeddings with BM25 lexical search to select the 20 closest candidate jobs from 361 indexed CareerViet postings.',
      color: 'from-green-500 to-emerald-500',
      bgColor: 'bg-green-50',
      textColor: 'text-green-600',
    },
    {
      icon: ChartBarIcon,
      title: '7-Dimension V3.1 Rule Scoring',
      description:
        'Evaluates skills, job title, experience, education, industry domain, location, and salary, while explicitly marking dimensions with insufficient evidence as unscored.',
      color: 'from-orange-500 to-red-500',
      bgColor: 'bg-orange-50',
      textColor: 'text-orange-600',
    },
    {
      icon: DocumentTextIcon,
      title: 'Hybrid TF-IDF + Rule Ranking',
      description:
        'The hybrid_qwen_v1 ranker blends the 7-dimension V3.1 rule score with Qwen3 embedding similarity and TF-IDF lexical overlap to produce a calibrated 0–100 fit score.',
      color: 'from-indigo-500 to-purple-500',
      bgColor: 'bg-indigo-50',
      textColor: 'text-indigo-600',
    },
    {
      icon: ChatBubbleBottomCenterTextIcon,
      title: 'Skill-by-Skill Evidence & Quotes',
      description:
        'Every ranked job card lists matched skills, missing required skills, plain-English explanations, and supporting quotes from the CV and job description.',
      color: 'from-teal-500 to-blue-500',
      bgColor: 'bg-teal-50',
      textColor: 'text-teal-600',
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.45,
      },
    },
  };

  return (
    <section className="py-20 bg-gradient-to-b from-white to-secondary-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center space-x-2 bg-primary-100 text-primary-800 px-4 py-2 rounded-full text-sm font-medium mb-4">
            <CpuChipIcon className="w-4 h-4" />
            <span>System Capabilities</span>
          </div>

          <h2 className="text-3xl md:text-4xl font-bold text-secondary-900 mb-4">
            Built for Transparent{' '}
            <span className="gradient-text">CV-to-Job Matching</span>
          </h2>

          <p className="text-lg text-secondary-600 max-w-3xl mx-auto">
            Instead of opaque keyword counts, every stage of our pipeline—from
            schema-validated extraction to 7-axis scoring—shows its reasoning
            and supporting text.
          </p>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
        >
          {features.map((feature) => (
            <motion.div
              key={feature.title}
              variants={itemVariants}
              className="group relative bg-white rounded-2xl p-7 shadow-soft hover:shadow-medium transition-all duration-300 border border-secondary-100 hover:border-primary-200"
            >
              <div
                className={`w-12 h-12 ${feature.bgColor} rounded-xl flex items-center justify-center mb-5 group-hover:scale-105 transition-transform duration-200`}
              >
                <feature.icon className={`w-6 h-6 ${feature.textColor}`} />
              </div>

              <h3 className="text-lg font-bold text-secondary-900 mb-2.5 group-hover:text-primary-600 transition-colors duration-200">
                {feature.title}
              </h3>

              <p className="text-sm text-secondary-600 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Features;
