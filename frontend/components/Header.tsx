import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import {
  Bars3Icon,
  XMarkIcon,
  DocumentTextIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import { getHealth, HealthResponse } from '../services/api';

type HealthState = 'loading' | 'ok' | 'degraded' | 'offline';

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [healthState, setHealthState] = useState<HealthState>('loading');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const router = useRouter();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then((data) => {
        if (!mounted) return;
        setHealthData(data);
        if (data.status === 'ok') {
          setHealthState('ok');
        } else {
          setHealthState('degraded');
        }
      })
      .catch(() => {
        if (!mounted) return;
        setHealthState('offline');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const navigation = [
    { name: 'Home', href: '/' },
    { name: 'Match CV', href: '/analyze' },
    { name: 'About', href: '/about' },
    { name: 'Help', href: '/help' },
  ];

  const isActive = (path: string) => router.pathname === path;

  const renderHealthPill = () => {
    if (healthState === 'loading') {
      return (
        <div
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-secondary-100 text-secondary-700 border border-secondary-200"
          role="status"
        >
          <span className="w-2 h-2 rounded-full bg-secondary-400 animate-pulse" />
          <span>Checking API…</span>
        </div>
      );
    }

    if (healthState === 'ok') {
      const jobCount = healthData?.jobs ?? 361;
      return (
        <div
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-success-50 text-success-800 border border-success-200"
          role="status"
        >
          <span className="w-2 h-2 rounded-full bg-success-600" />
          <span>{jobCount} jobs ready</span>
        </div>
      );
    }

    if (healthState === 'degraded') {
      return (
        <div
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning-50 text-warning-800 border border-warning-200"
          role="status"
        >
          <span className="w-2 h-2 rounded-full bg-warning-600" />
          <span>Extractor offline — cached CVs only</span>
        </div>
      );
    }

    return (
      <div
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-error-50 text-error-800 border border-error-200"
        role="status"
      >
        <span className="w-2 h-2 rounded-full bg-error-600" />
        <span>API offline</span>
      </div>
    );
  };

  return (
    <header
      className={`fixed w-full z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-white/95 backdrop-blur-md shadow-soft border-b border-secondary-100'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4 gap-3">
          {/* Logo */}
          <Link
            href="/"
            className="flex items-center space-x-2 group focus:outline-none focus:ring-2 focus:ring-primary-500 rounded-lg"
          >
            <div className="relative">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-accent-600 rounded-xl flex items-center justify-center transform group-hover:scale-105 transition-transform duration-200">
                <DocumentTextIcon className="w-6 h-6 text-white" />
              </div>
              <SparklesIcon className="w-4 h-4 text-accent-500 absolute -top-1 -right-1 animate-bounce-subtle" />
            </div>
            <div>
              <span className="text-xl font-bold gradient-text">ResuMatch</span>
              <span className="block text-xs text-secondary-500 -mt-1">
                CareerViet Matcher
              </span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-6">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={`relative px-3 py-2 text-sm font-medium transition-colors duration-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                  isActive(item.href)
                    ? 'text-primary-600'
                    : 'text-secondary-600 hover:text-secondary-900'
                }`}
              >
                {item.name}
                {isActive(item.href) && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 rounded-full"
                    initial={false}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  />
                )}
              </Link>
            ))}
          </nav>

          {/* Service Health Pill + CTA Button */}
          <div className="hidden md:flex items-center space-x-3">
            {renderHealthPill()}
            <Link
              href="/analyze"
              className="btn-primary inline-flex items-center space-x-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <SparklesIcon className="w-4 h-4" />
              <span>Match CV</span>
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            type="button"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={isMenuOpen}
            className="md:hidden p-2 rounded-lg text-secondary-600 hover:text-secondary-900 hover:bg-secondary-100 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {isMenuOpen ? (
              <XMarkIcon className="w-6 h-6" />
            ) : (
              <Bars3Icon className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="md:hidden py-4 border-t border-secondary-200 bg-white/95 backdrop-blur-md rounded-b-xl px-2"
          >
            <div className="flex flex-col space-y-2">
              <div className="px-3 py-1">{renderHealthPill()}</div>
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsMenuOpen(false)}
                  className={`px-3 py-2 text-base font-medium rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 ${
                    isActive(item.href)
                      ? 'text-primary-600 bg-primary-50'
                      : 'text-secondary-600 hover:text-secondary-900 hover:bg-secondary-50'
                  }`}
                >
                  {item.name}
                </Link>
              ))}
              <div className="pt-2 px-3">
                <Link
                  href="/analyze"
                  onClick={() => setIsMenuOpen(false)}
                  className="btn-primary w-full justify-center inline-flex items-center space-x-2 text-sm"
                >
                  <SparklesIcon className="w-4 h-4" />
                  <span>Match CV</span>
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </header>
  );
};

export default Header;
