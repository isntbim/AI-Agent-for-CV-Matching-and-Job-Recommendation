import React from 'react';
import Link from 'next/link';
import { ArrowRightIcon } from '@heroicons/react/24/outline';
import { BentoLogoMark } from './Header';

interface FooterProps {
  hideClosingCta?: boolean;
}

const Footer: React.FC<FooterProps> = ({ hideClosingCta = false }) => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-surface border-t border-hairline mt-auto">
      <div className="wrap py-12 sm:py-16 space-y-12">
        {/* Row 1: Closing Action Card */}
        {!hideClosingCta && (
          <div className="bento-card bento-card--highlight p-6 sm:p-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="max-w-2xl space-y-2">
              <h2 className="text-[28px] sm:text-[36px] leading-[1.14] text-ink">
                Rank your CV against 361 job postings, or paste your own.
              </h2>
              <p className="text-[16px] text-body">
                Run a PDF, DOCX, or TXT resume through three models in about
                14&nbsp;GB of GPU memory and
                inspect every dimension score and verbatim quote.
              </p>
            </div>
            <div className="shrink-0">
              <Link href="/analyze" className="btn-primary w-full sm:w-auto">
                <span>Upload a CV</span>
                <ArrowRightIcon
                  className="w-4 h-4 m-arrow"
                  aria-hidden="true"
                />
              </Link>
            </div>
          </div>
        )}

        {/* Row 2: Directory (44px tap targets on every link) */}
        <div className="grid grid-cols-2 sm:grid-cols-12 gap-6 sm:gap-8">
          <div className="col-span-2 sm:col-span-12 lg:col-span-6 space-y-3 min-w-0">
            <Link
              href="/"
              className="inline-flex items-center gap-2.5 min-h-[44px] text-ink no-underline"
            >
              <BentoLogoMark className="w-7 h-7" />
              <span className="text-[20px] font-medium tracking-[-0.03em] text-ink">
                Matchly
              </span>
            </Link>
            <p className="text-[16px] text-muted max-w-[52ch] leading-relaxed">
              CV-to-job matching over 361 public job postings crawled from CareerViet and ITviec:
              QLoRA Qwen2.5-7B extraction, hybrid BGE-M3 and BM25 retrieval,
              seven weighted V3.1 rules, and Qwen3-Embedding-4B ranking.
            </p>
          </div>

          <div className="col-span-1 sm:col-span-4 lg:col-span-3 space-y-1 min-w-0">
            <h3 className="text-[15px] font-medium text-ink pb-1">Pages</h3>
            <ul className="text-[15px]">
              <li>
                <Link
                  href="/"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Overview
                </Link>
              </li>
              <li>
                <Link
                  href="/analyze"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Upload a CV
                </Link>
              </li>
              <li>
                <Link
                  href="/about"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Architecture
                </Link>
              </li>
              <li>
                <Link
                  href="/help"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Usage guide
                </Link>
              </li>
            </ul>
          </div>

          <div className="col-span-1 sm:col-span-4 lg:col-span-3 space-y-1 min-w-0">
            <h3 className="text-[15px] font-medium text-ink pb-1">
              Specification
            </h3>
            <ul className="text-[15px]">
              <li>
                <Link
                  href="/about#pipeline"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Four-stage pipeline
                </Link>
              </li>
              <li>
                <Link
                  href="/about#weights"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Seven rule weights
                </Link>
              </li>
              <li>
                <Link
                  href="/about#runtime"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  14 GB GPU runtime
                </Link>
              </li>
              <li>
                <Link
                  href="/help#faq"
                  className="inline-flex items-center min-h-[44px] text-body hover:text-ink underline-offset-4 hover:underline"
                >
                  Common questions
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Row 3: Legal & MIT Attribution (NOTICE.md) */}
        <div className="pt-6 border-t border-hairline flex flex-col md:flex-row md:items-center justify-between gap-3 text-[13px] text-muted">
          <p>
            © {currentYear} ATS CV-JD Matching Capstone (FA26AI45). Models served
            by Ollama on a Runpod serverless GPU, behind FastAPI. Job postings were
            collected from the public pages of CareerViet and ITviec; Matchly is not
            affiliated with either site.
          </p>
          <p>
            Frontend adapted from{' '}
            <a
              href="https://github.com/gaurav-singh7092/ResuMatch"
              target="_blank"
              rel="noopener noreferrer"
              className="text-link inline-flex items-center min-h-[44px] sm:min-h-0"
            >
              ResuMatch
            </a>{' '}
            by Gaurav Singh (MIT License, kept in{' '}
            <span className="font-mono">frontend/NOTICE.md</span>).
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
