import React from 'react';
import Link from 'next/link';
import {
  DocumentTextIcon,
  SparklesIcon,
  CodeBracketIcon,
} from '@heroicons/react/24/outline';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    navigation: [
      { name: 'Home', href: '/' },
      { name: 'Match Your CV', href: '/analyze' },
      { name: 'How It Works', href: '/about' },
      { name: 'Help & FAQ', href: '/help' },
    ],
    pipeline: [
      { name: 'Qwen2.5-7B Extractor (QLoRA)', href: '/about#pipeline' },
      { name: 'Hybrid BGE-M3 + BM25 Search', href: '/about#pipeline' },
      { name: '7-Dimension Rule + Hybrid Scorer', href: '/about#scoring' },
      { name: 'Troubleshooting Guide', href: '/help#faq' },
    ],
  };

  return (
    <footer className="bg-secondary-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="lg:col-span-2">
            <Link
              href="/"
              className="inline-flex items-center space-x-2 group mb-4 focus:outline-none focus:ring-2 focus:ring-primary-400 rounded-lg"
            >
              <div className="relative">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl flex items-center justify-center transform group-hover:scale-105 transition-transform duration-200">
                  <DocumentTextIcon className="w-6 h-6 text-white" />
                </div>
                <SparklesIcon className="w-4 h-4 text-accent-400 absolute -top-1 -right-1" />
              </div>
              <div>
                <span className="text-xl font-bold text-white">ResuMatch</span>
                <span className="block text-xs text-secondary-400 -mt-1">
                  CareerViet CV-to-Jobs Matcher
                </span>
              </div>
            </Link>

            <p className="text-secondary-300 text-sm mb-6 max-w-md leading-relaxed">
              Upload your CV once to match against 361 CareerViet job postings.
              Powered by a fine-tuned Qwen2.5-7B extractor with LangGraph
              validate-repair-retry, hybrid BGE-M3 + BM25 retrieval, and
              7-dimension explainable fit scoring.
            </p>
          </div>

          {/* Navigation Links */}
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">
              Navigation
            </h3>
            <ul className="space-y-3">
              {footerLinks.navigation.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-secondary-300 hover:text-white text-sm transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-400 rounded"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Pipeline Links */}
          <div>
            <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-4">
              System Architecture
            </h3>
            <ul className="space-y-3">
              {footerLinks.pipeline.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-secondary-300 hover:text-white text-sm transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-400 rounded"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="mt-10 pt-8 border-t border-secondary-800 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="text-secondary-400 text-sm">
            © {currentYear} ATS CV-JD Matching Capstone (FA26AI45). Frontend adapted from ResuMatch (MIT, per its
            README).
          </div>

          <div className="flex items-center space-x-2 text-secondary-400 text-sm">
            <CodeBracketIcon className="w-4 h-4 text-primary-400" />
            <span>
              <a
                href="https://github.com/gaurav-singh7092/ResuMatch"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-400 hover:text-primary-300 underline focus:outline-none focus:ring-2 focus:ring-primary-400 rounded"
              >
                Based on ResuMatch
              </a>{' '}
              by Gaurav Singh
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
