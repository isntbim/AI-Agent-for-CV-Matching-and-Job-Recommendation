import React, { Component, ReactNode } from 'react';
import { ExclamationTriangleIcon, ArrowPathIcon } from '@heroicons/react/24/outline';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-ground flex items-center justify-center p-6">
          <div className="max-w-lg w-full bento-card p-8 space-y-5">
            <div className="w-11 h-11 rounded-ctrl border border-bad-line bg-bad-bg flex items-center justify-center text-bad">
              <ExclamationTriangleIcon className="w-6 h-6" aria-hidden="true" />
            </div>
            <div className="space-y-2">
              <h1 className="text-[24px] text-ink">
                Interface error encountered
              </h1>
              <p className="text-[16px] text-body">
                A component failed to render. Retry the view or reload the
                application.
              </p>
            </div>
            {this.state.error && (
              <details className="text-left">
                <summary className="cursor-pointer text-[14px] text-muted py-1">
                  Inspect error trace
                </summary>
                <pre className="mt-2 text-[13px] font-mono bg-surface p-3 rounded-ctrl border border-hairline overflow-auto max-h-36 text-ink">
                  {this.state.error.toString()}
                </pre>
              </details>
            )}
            <div className="flex flex-col sm:flex-row gap-3 pt-1">
              <button
                type="button"
                onClick={this.handleReset}
                className="btn-primary"
              >
                <ArrowPathIcon className="w-4 h-4" aria-hidden="true" />
                <span>Retry view</span>
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="btn-secondary"
              >
                <span>Reload page</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
