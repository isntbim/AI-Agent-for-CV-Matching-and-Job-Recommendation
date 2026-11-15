import React, { useState, useEffect, useRef, useCallback, useId } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import {
  Bars3Icon,
  XMarkIcon,
  SunIcon,
  MoonIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';
import { getHealth, HealthResponse } from '../services/api';

type HealthState = 'loading' | 'ok' | 'degraded' | 'offline';

const NAVIGATION = [
  { name: 'Overview', href: '/' },
  { name: 'Matcher', href: '/analyze' },
  { name: 'Architecture', href: '/about' },
  { name: 'Guide', href: '/help' },
];

export const BentoLogoMark: React.FC<{ className?: string }> = ({
  className = 'w-7 h-7',
}) => {
  const rawId = useId();
  const gradId = `bento-logo-${rawId.replace(/:/g, '')}`;
  return (
    <svg
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <defs>
        <linearGradient
          id={gradId}
          x1="19"
          y1="6"
          x2="26"
          y2="15"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="var(--ring-stop-1)" />
          <stop offset="100%" stopColor="var(--ring-stop-3)" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="var(--ink)" />
      <rect x="6" y="6" width="11" height="9" rx="2.5" fill="var(--bg)" />
      <rect x="19" y="6" width="7" height="9" rx="2.5" fill={`url(#${gradId})`} />
      <rect
        x="6"
        y="17"
        width="7"
        height="9"
        rx="2.5"
        fill="var(--bg)"
        fillOpacity="0.58"
      />
      <rect x="15" y="17" width="11" height="9" rx="2.5" fill="var(--bg)" />
    </svg>
  );
};

const Header: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isDark, setIsDark] = useState(false);
  const [healthState, setHealthState] = useState<HealthState>('loading');
  const [healthData, setHealthData] = useState<HealthResponse | null>(null);
  const router = useRouter();

  const topBarRef = useRef<HTMLDivElement | null>(null);
  const menuToggleRef = useRef<HTMLButtonElement | null>(null);
  const mobileSheetRef = useRef<HTMLDivElement | null>(null);
  const dialogCloseRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const syncTheme = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    syncTheme();
    const observer = new MutationObserver(syncTheme);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    return () => observer.disconnect();
  }, []);

  const closeMenu = useCallback((restoreFocus = false) => {
    setIsMenuOpen(false);
    if (restoreFocus) {
      requestAnimationFrame(() => {
        menuToggleRef.current?.focus();
      });
    }
  }, []);

  // Close mobile menu automatically on client route changes
  useEffect(() => {
    const handleRouteDone = () => setIsMenuOpen(false);
    router.events.on('routeChangeComplete', handleRouteDone);
    return () => {
      router.events.off('routeChangeComplete', handleRouteDone);
    };
  }, [router.events]);

  // Modal focus containment strictly inside role="dialog", background inertness, and lg resize auto-close
  useEffect(() => {
    if (!isMenuOpen) return;

    const desktopMq = window.matchMedia('(min-width: 1024px)');
    if (desktopMq.matches) {
      setIsMenuOpen(false);
      return;
    }

    const onBreakpointChange = (ev: MediaQueryListEvent) => {
      if (ev.matches) {
        setIsMenuOpen(false);
      }
    };
    desktopMq.addEventListener('change', onBreakpointChange);

    const backgroundNodes: HTMLElement[] = [
      ...Array.from(document.querySelectorAll<HTMLElement>('main, footer')),
      ...(topBarRef.current ? [topBarRef.current] : []),
    ];
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    for (const el of backgroundNodes) {
      el.setAttribute('inert', '');
      el.setAttribute('aria-hidden', 'true');
    }

    dialogCloseRef.current?.focus();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        closeMenu(true);
        return;
      }

      if (e.key === 'Tab' && mobileSheetRef.current) {
        const focusables = Array.from(
          mobileSheetRef.current.querySelectorAll<HTMLElement>(
            'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
          )
        ).filter((el) => el.offsetParent !== null);

        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;

        if (
          e.shiftKey &&
          (active === first || !mobileSheetRef.current.contains(active))
        ) {
          e.preventDefault();
          last.focus();
        } else if (
          !e.shiftKey &&
          (active === last || !mobileSheetRef.current.contains(active))
        ) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    window.addEventListener('keydown', onKeyDown);
    return () => {
      desktopMq.removeEventListener('change', onBreakpointChange);
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = prevOverflow;
      for (const el of backgroundNodes) {
        el.removeAttribute('inert');
        el.removeAttribute('aria-hidden');
      }
    };
  }, [isMenuOpen, closeMenu]);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then((data) => {
        if (!mounted) return;
        setHealthData(data);
        setHealthState(data.status === 'ok' ? 'ok' : 'degraded');
      })
      .catch(() => {
        if (!mounted) return;
        setHealthState('offline');
      });
    return () => {
      mounted = false;
    };
  }, []);

  const toggleTheme = () => {
    const nextDark = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', nextDark);
    setIsDark(nextDark);
    try {
      localStorage.setItem('resumatch-theme', nextDark ? 'dark' : 'light');
    } catch {
      // Ignore storage errors in restricted contexts
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute('content', nextDark ? '#070d1a' : '#f4f7fc');
    }
  };

  const isActive = (path: string) => router.pathname === path;

  const renderHealthStatus = () => {
    if (healthState === 'loading') {
      return (
        <div
          role="status"
          className="inline-flex items-center gap-2 px-2.5 py-1 rounded-ctrl border border-hairline bg-ground text-[13px] font-mono text-muted"
        >
          <span className="w-2 h-2 rounded-full bg-muted" />
          <span>Checking API…</span>
        </div>
      );
    }

    if (healthState === 'ok') {
      const jobs = healthData?.jobs ?? 361;
      return (
        <div
          role="status"
          className="inline-flex items-center gap-2 px-2.5 py-1 rounded-ctrl border border-good-line bg-good-bg text-[13px] font-mono text-good"
        >
          <span className="w-2 h-2 rounded-full bg-good" />
          <span>{jobs} jobs · Qwen up</span>
        </div>
      );
    }

    if (healthState === 'degraded') {
      return (
        <div
          role="status"
          title="Matching unavailable while Ollama is stopped; saved runs remain available."
          className="inline-flex items-center gap-2 px-2.5 py-1 rounded-ctrl border border-pot-line bg-pot-bg text-[13px] font-mono text-pot"
        >
          <span className="w-2 h-2 rounded-full bg-pot" />
          <span>Matching unavailable · saved runs ok</span>
        </div>
      );
    }

    return (
      <div
        role="status"
        className="inline-flex items-center gap-2 px-2.5 py-1 rounded-ctrl border border-bad-line bg-bad-bg text-[13px] font-mono text-bad"
      >
        <span className="w-2 h-2 rounded-full bg-bad" />
        <span>API offline</span>
      </div>
    );
  };

  return (
    <header className="fixed top-0 inset-x-0 z-40 h-[56px] sm:h-[72px] bg-card border-b border-hairline">
      <div
        ref={topBarRef}
        className="wrap relative h-full flex items-center justify-between gap-3"
      >
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>

        {/* Brand */}
        <Link
          href="/"
          className="inline-flex items-center gap-2.5 min-h-[44px] rounded-ctrl pr-1 text-ink no-underline min-w-0"
        >
          <BentoLogoMark className="w-7 h-7 shrink-0" />
          <span className="text-[18px] sm:text-[20px] font-medium tracking-[-0.03em] text-ink truncate">
            ResuMatch
          </span>
        </Link>

        {/* Desktop Centre Nav */}
        <nav
          aria-label="Primary"
          className="hidden lg:flex items-center gap-7"
        >
          {NAVIGATION.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={`relative inline-flex items-center min-h-[44px] text-[15px] font-medium whitespace-nowrap no-underline transition-colors ${
                  active ? 'text-ink' : 'text-muted hover:text-ink'
                }`}
              >
                <span>{item.name}</span>
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute bottom-[6px] left-0 right-0 h-[2px] bar-brand rounded-full"
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Actions */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <div className="hidden xl:block">{renderHealthStatus()}</div>

          <button
            type="button"
            onClick={toggleTheme}
            aria-label={
              isDark ? 'Switch to light theme' : 'Switch to dark theme'
            }
            title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
            className="inline-flex items-center justify-center w-[44px] h-[44px] rounded-ctrl border border-hairline bg-ground text-ink hover:bg-surface transition-colors shrink-0"
          >
            {isDark ? (
              <SunIcon className="w-5 h-5" aria-hidden="true" />
            ) : (
              <MoonIcon className="w-5 h-5" aria-hidden="true" />
            )}
          </button>

          <div className="hidden sm:block">
            <Link href="/analyze" className="btn-primary text-[15px] px-4">
              <span>Upload a CV</span>
              <ArrowRightIcon className="w-4 h-4 m-arrow" aria-hidden="true" />
            </Link>
          </div>

          <button
            ref={menuToggleRef}
            type="button"
            onClick={() => setIsMenuOpen(true)}
            aria-label="Open navigation menu"
            aria-expanded={isMenuOpen}
            aria-controls="mobile-nav-sheet"
            className="lg:hidden inline-flex items-center justify-center w-[44px] h-[44px] rounded-ctrl border border-hairline bg-ground text-ink hover:bg-surface transition-colors shrink-0"
          >
            <Bars3Icon className="w-5 h-5" aria-hidden="true" />
          </button>
        </div>
      </div>

      {/* Fullscreen Mobile / Tablet Modal Dialog (self-contained focus boundary) */}
      {isMenuOpen && (
        <div
          id="mobile-nav-sheet"
          ref={mobileSheetRef}
          role="dialog"
          aria-modal="true"
          aria-label="Navigation menu"
          className="lg:hidden fixed inset-0 bg-card overflow-y-auto z-50 flex flex-col"
        >
          <div className="h-[56px] sm:h-[72px] border-b border-hairline shrink-0">
            <div className="wrap h-full flex items-center justify-between gap-3">
              <Link
                href="/"
                onClick={() => closeMenu(false)}
                className="inline-flex items-center gap-2.5 min-h-[44px] rounded-ctrl pr-1 text-ink no-underline min-w-0"
              >
                <BentoLogoMark className="w-7 h-7 shrink-0" />
                <span className="text-[18px] sm:text-[20px] font-medium tracking-[-0.03em] text-ink truncate">
                  ResuMatch
                </span>
              </Link>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={toggleTheme}
                  aria-label={
                    isDark ? 'Switch to light theme' : 'Switch to dark theme'
                  }
                  className="inline-flex items-center justify-center w-[44px] h-[44px] rounded-ctrl border border-hairline bg-ground text-ink hover:bg-surface transition-colors"
                >
                  {isDark ? (
                    <SunIcon className="w-5 h-5" aria-hidden="true" />
                  ) : (
                    <MoonIcon className="w-5 h-5" aria-hidden="true" />
                  )}
                </button>

                <button
                  ref={dialogCloseRef}
                  type="button"
                  onClick={() => closeMenu(true)}
                  aria-label="Close navigation menu"
                  className="inline-flex items-center justify-center w-[44px] h-[44px] rounded-ctrl border border-hairline bg-ground text-ink hover:bg-surface transition-colors"
                >
                  <XMarkIcon className="w-5 h-5" aria-hidden="true" />
                </button>
              </div>
            </div>
          </div>

          <div className="wrap py-6 flex-1 flex flex-col justify-between gap-6">
            <nav aria-label="Mobile" className="divide-y divide-hairline">
              {NAVIGATION.map((item) => {
                const active = isActive(item.href);
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    aria-current={active ? 'page' : undefined}
                    onClick={() => closeMenu(false)}
                    className={`flex items-center justify-between min-h-[56px] text-[17px] font-medium no-underline ${
                      active ? 'text-ink' : 'text-muted hover:text-ink'
                    }`}
                  >
                    <span>{item.name}</span>
                    {active && (
                      <span className="text-[13px] font-mono text-accent-text">
                        Current
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>

            <div className="space-y-4 pb-4">
              <div>{renderHealthStatus()}</div>
              <Link
                href="/analyze"
                onClick={() => closeMenu(false)}
                className="btn-primary w-full"
              >
                <span>Upload a CV</span>
                <ArrowRightIcon
                  className="w-4 h-4 m-arrow"
                  aria-hidden="true"
                />
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
