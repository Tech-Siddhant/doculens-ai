"use client";

import React from "react";
import { AlertCircle, RefreshCw, X } from "lucide-react";
import { Button } from "./Button";

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  className?: string;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = "Something went wrong",
  message,
  onRetry,
  onDismiss,
  className = "",
}) => {
  return (
    <div
      role="alert"
      className={`flex items-start gap-3 p-4 bg-rose-50 border border-rose-200 rounded text-rose-950 ${className}`}
    >
      <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <h4 className="text-sm font-semibold text-rose-900">{title}</h4>
        <p className="text-sm text-rose-800 mt-0.5 leading-relaxed">{message}</p>
        {onRetry && (
          <div className="mt-3">
            <Button
              size="sm"
              variant="outline"
              onClick={onRetry}
              leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
              className="bg-white/80 border-rose-300 text-rose-900 hover:bg-white"
            >
              Try again
            </Button>
          </div>
        )}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          className="text-rose-500 hover:text-rose-700 p-1 rounded hover:bg-rose-100 transition-colors"
          aria-label="Dismiss alert"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};
