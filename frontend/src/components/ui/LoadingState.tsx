import React from "react";
import { Loader2 } from "lucide-react";

interface LoadingStateProps {
  message?: string;
  description?: string;
  className?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = "Loading...",
  description,
  className = "",
}) => {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex flex-col items-center justify-center p-8 text-center ${className}`}
    >
      <Loader2 className="w-8 h-8 animate-spin text-brand mb-3" />
      <p className="text-sm font-medium text-text-primary">{message}</p>
      {description && (
        <p className="text-xs text-text-secondary mt-1 max-w-sm">{description}</p>
      )}
      <span className="sr-only">Loading content</span>
    </div>
  );
};

export const SkeletonCard: React.FC<{ className?: string }> = ({
  className = "",
}) => {
  return (
    <div
      className={`bg-surface-elevated border border-border-subtle p-5 rounded animate-pulse space-y-3 ${className}`}
    >
      <div className="h-4 bg-surface-container rounded w-1/3" />
      <div className="h-3 bg-surface-container rounded w-3/4" />
      <div className="h-3 bg-surface-container rounded w-1/2" />
    </div>
  );
};

export const SkeletonTable: React.FC<{ rows?: number; className?: string }> = ({
  rows = 4,
  className = "",
}) => {
  return (
    <div
      className={`bg-surface-elevated border border-border-subtle rounded overflow-hidden ${className}`}
      role="status"
      aria-label="Loading documents"
    >
      <div className="bg-surface-low/50 border-b border-border-subtle py-3.5 px-4 flex items-center justify-between">
        <div className="h-3 bg-surface-container rounded w-28" />
        <div className="hidden sm:flex gap-8">
          <div className="h-3 bg-surface-container rounded w-16" />
          <div className="h-3 bg-surface-container rounded w-20" />
          <div className="h-3 bg-surface-container rounded w-20" />
        </div>
        <div className="h-3 bg-surface-container rounded w-12" />
      </div>
      <div className="divide-y divide-border-subtle">
        {Array.from({ length: rows }).map((_, i) => (
          <div
            key={i}
            className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 animate-pulse"
          >
            <div className="flex items-center gap-3 w-full sm:w-1/3">
              <div className="w-9 h-9 rounded bg-surface-container shrink-0" />
              <div className="space-y-1.5 flex-1 min-w-0">
                <div className="h-3.5 bg-surface-container rounded w-3/4" />
                <div className="h-2.5 bg-surface-container rounded w-1/2" />
              </div>
            </div>
            <div className="hidden sm:block h-3 bg-surface-container rounded w-16" />
            <div className="h-5 bg-surface-container rounded w-24" />
            <div className="hidden sm:block h-3 bg-surface-container rounded w-20" />
            <div className="h-7 bg-surface-container rounded w-20 self-end sm:self-auto" />
          </div>
        ))}
      </div>
      <span className="sr-only">Loading documents table</span>
    </div>
  );
};
