import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SplitText } from 'gsap/SplitText';
import { useGSAP } from '@gsap/react';

let isRegistered = false;

export const MOTION_QUERIES = {
  canAnimate: '(prefers-reduced-motion: no-preference)',
  reduceMotion: '(prefers-reduced-motion: reduce)',
} as const;

/**
 * Registers GSAP plugins once on the client. Safe to import in Next.js SSR.
 */
export function ensureGsapRegistered(): void {
  if (typeof window === 'undefined' || isRegistered) return;
  gsap.registerPlugin(useGSAP, ScrollTrigger, SplitText);
  gsap.defaults({
    duration: 0.6,
    ease: 'power2.out',
  });
  isRegistered = true;
}

if (typeof window !== 'undefined') {
  ensureGsapRegistered();
}

export { gsap, ScrollTrigger, SplitText, useGSAP };
