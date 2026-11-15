import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export default function LoadingSpinner({
  size = 'md',
  className = '',
}: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-9 h-9 border-[2.5px]',
  };

  return (
    <span
      role="status"
      aria-label="Loading"
      className={`inline-block rounded-full border-hairline border-t-accent animate-spin ${sizeClasses[size]} ${className}`}
    />
  );
}
