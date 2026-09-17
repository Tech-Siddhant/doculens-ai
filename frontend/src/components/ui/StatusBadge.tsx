import React from "react";
import { DocumentStatus } from "@/types";
import { CheckCircle2, Clock, AlertTriangle, XCircle } from "lucide-react";

interface StatusBadgeProps {
  status: DocumentStatus | string;
  size?: "sm" | "md";
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = "md",
  className = "",
}) => {
  const normalizedStatus = status.toLowerCase().replace(/[\s-]/g, "_");

  switch (normalizedStatus) {
    case "ready":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded border ${
            size === "sm" ? "text-xs px-2 py-0.5" : "text-xs px-2.5 py-1"
          } bg-emerald-50 border-emerald-200 text-emerald-800 ${className}`}
        >
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>Ready</span>
        </span>
      );

    case "processing":
    case "indexing":
    case "extracting":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded border ${
            size === "sm" ? "text-xs px-2 py-0.5" : "text-xs px-2.5 py-1"
          } bg-amber-50 border-amber-200 text-amber-800 ${className}`}
        >
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
          </span>
          <Clock className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>Processing</span>
        </span>
      );

    case "needs_attention":
    case "degraded":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded border ${
            size === "sm" ? "text-xs px-2 py-0.5" : "text-xs px-2.5 py-1"
          } bg-amber-50 border-amber-300 text-amber-900 ${className}`}
        >
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>Needs attention</span>
        </span>
      );

    case "failed":
    case "error":
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded border ${
            size === "sm" ? "text-xs px-2 py-0.5" : "text-xs px-2.5 py-1"
          } bg-rose-50 border-rose-200 text-rose-800 ${className}`}
        >
          <XCircle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
          <span>Failed</span>
        </span>
      );

    default:
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded border ${
            size === "sm" ? "text-xs px-2 py-0.5" : "text-xs px-2.5 py-1"
          } bg-slate-50 border-slate-200 text-slate-700 ${className}`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
          <span className="capitalize">{status}</span>
        </span>
      );
  }
};
