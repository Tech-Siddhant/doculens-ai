"use client";

import React, { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Info,
  Maximize2,
  Sliders,
  Layers,
  Database,
  Search,
  Sparkles,
  ShieldCheck,
} from "lucide-react";
import { PipelineTrace, PipelineStageTrace, Citation } from "@/types";

const STAGE_EXPLANATIONS: Record<string, { icon: React.FC<{ className?: string }>; description: string; tooltip: string }> = {
  question_understanding: {
    icon: Search,
    description: "Sanitizes and normalizes the user query for search indexing.",
    tooltip: "Preprocesses the question, stripping whitespace and determining query boundaries.",
  },
  hybrid_retrieval: {
    icon: Database,
    description: "Multi-channel retrieval across vector embeddings, BM25 keywords, and visual pages.",
    tooltip: "Retrieves candidate passages simultaneously using dense semantic vectors, sparse BM25 lexical search, and visual page layout.",
  },
  reranking: {
    icon: Sliders,
    description: "Cross-encoder scoring — prioritizes the most relevant retrieved passages.",
    tooltip: "Scores (question, passage) pairs with a neural cross-encoder model to re-order the most relevant evidence.",
  },
  evidence_selection: {
    icon: Layers,
    description: "Filters, deduplicates, and caps the highest-ranking passages.",
    tooltip: "Applies score thresholds and deduplication across chunk IDs and page numbers to select top-K evidence.",
  },
  evidence_validation: {
    icon: ShieldCheck,
    description: "Verifies metadata and structural completeness of selected evidence chunks.",
    tooltip: "Ensures evidence items contain valid text snippets or visual page references and strictly belong to the target document.",
  },
  context_assembly: {
    icon: Layers,
    description: "Packages validated evidence into an isolated prompt block while defending against prompt injection.",
    tooltip: "Formats evidence tags with strict XML escaping and size bounds to prevent prompt breakout attacks.",
  },
  answer_generation: {
    icon: Sparkles,
    description: "Synthesizes a grounded answer from assembled evidence using the selected LLM.",
    tooltip: "Directs the language model to answer solely based on provided evidence with strict grounding rules.",
  },
  citation_validation: {
    icon: ShieldCheck,
    description: "Verifies that every cited claim maps to an actual retrieved evidence chunk.",
    tooltip: "Validates bracketed references (e.g. [Evidence 1]) against server-side evidence items and drops hallucinated citations.",
  },
};

const SENSITIVE_KEYS = new Set([
  "api_key",
  "apikey",
  "secret",
  "token",
  "password",
  "system_prompt",
  "prompt",
  "raw_prompt",
  "hidden_reasoning",
]);

interface PipelineTimelineProps {
  trace?: PipelineTrace | null;
  citations?: Citation[];
  onNavigateToPage?: (pageNumber: number, snippet?: string, citation?: Citation) => void;
  onInspectEvidence?: (citation?: Citation) => void;
}

export const PipelineTimeline: React.FC<PipelineTimelineProps> = ({
  trace,
  citations = [],
  onNavigateToPage,
  onInspectEvidence,
}) => {
  const [expandAll, setExpandAll] = useState(false);
  const [expandedStages, setExpandedStages] = useState<Record<string, boolean>>({});

  if (!trace || !trace.stages || trace.stages.length === 0) {
    return (
      <div className="bg-surface-low border border-border-subtle rounded-lg p-3 sm:p-4 text-xs text-text-tertiary">
        No pipeline telemetry available for this query.
      </div>
    );
  }

  const toggleStage = (stageId: string) => {
    setExpandedStages((prev) => ({
      ...prev,
      [stageId]: !prev[stageId],
    }));
  };

  const handleToggleAll = () => {
    const nextState = !expandAll;
    setExpandAll(nextState);
    const newExpanded: Record<string, boolean> = {};
    trace.stages.forEach((s) => {
      newExpanded[s.stage_id] = nextState;
    });
    setExpandedStages(newExpanded);
  };

  const summary = trace.summary || {
    total_duration_ms: 0,
    total_stages: trace.stages.length,
    successful_stages: trace.stages.filter((s) => s.status === "success").length,
    failed_stages: trace.stages.filter((s) => s.status === "failed").length,
    fallback_stages: trace.stages.filter((s) => s.status === "fallback").length,
  };

  return (
    <div className="bg-surface-low border border-border-subtle rounded-lg p-3 sm:p-4 text-sm mt-3 mb-2 space-y-3.5 transition-all">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border-subtle pb-2.5">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded bg-brand-light text-brand">
            <Clock className="w-3.5 h-3.5" />
          </div>
          <div>
            <h4 className="font-heading font-semibold text-text-primary text-xs uppercase tracking-wider flex items-center gap-1.5">
              Pipeline Execution Trace
            </h4>
            <span className="text-[10px] text-text-tertiary">
              {summary.successful_stages} of {summary.total_stages} stages succeeded
              {summary.fallback_stages > 0 ? ` · ${summary.fallback_stages} fallback` : ""}
              {summary.failed_stages > 0 ? ` · ${summary.failed_stages} failed` : ""}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 self-end sm:self-auto">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-text-secondary bg-surface-container px-2 py-0.5 rounded border border-border-subtle">
            <span className="font-semibold text-text-primary">
              {(summary.total_duration_ms || 0).toFixed(0)} ms
            </span>
            {summary.retrieval_duration_ms ? (
              <span title="Retrieval Latency" className="text-text-tertiary">
                · Ret: {summary.retrieval_duration_ms.toFixed(0)}ms
              </span>
            ) : null}
            {summary.reranking_duration_ms ? (
              <span title="Reranking Latency" className="text-text-tertiary">
                · Rnk: {summary.reranking_duration_ms.toFixed(0)}ms
              </span>
            ) : null}
            {summary.generation_duration_ms ? (
              <span title="Generation Latency" className="text-text-tertiary">
                · Gen: {summary.generation_duration_ms.toFixed(0)}ms
              </span>
            ) : null}
            {summary.citation_validation_duration_ms ? (
              <span title="Citation Validation Latency" className="text-text-tertiary">
                · Cit: {summary.citation_validation_duration_ms.toFixed(0)}ms
              </span>
            ) : null}
          </div>

          <button
            type="button"
            onClick={handleToggleAll}
            className="text-[11px] font-medium text-brand hover:text-brand-dark transition-colors"
          >
            {expandAll ? "Collapse all" : "Expand all"}
          </button>
        </div>
      </div>

      <div className="space-y-1.5">
        {trace.stages.map((stage) => {
          const isExpanded =
            expandedStages[stage.stage_id] !== undefined
              ? expandedStages[stage.stage_id]
              : expandAll;

          return (
            <StageItem
              key={stage.stage_id}
              stage={stage}
              isExpanded={isExpanded}
              onToggle={() => toggleStage(stage.stage_id)}
              onNavigate={onNavigateToPage}
              onInspectEvidence={onInspectEvidence}
              citations={citations}
            />
          );
        })}
      </div>
    </div>
  );
};

interface StageItemProps {
  stage: PipelineStageTrace;
  isExpanded: boolean;
  onToggle: () => void;
  onNavigate?: (pageNumber: number, snippet?: string, citation?: Citation) => void;
  onInspectEvidence?: (citation?: Citation) => void;
  citations?: Citation[];
}

const StageItem: React.FC<StageItemProps> = ({
  stage,
  isExpanded,
  onToggle,
  onNavigate,
  onInspectEvidence,
  citations = [],
}) => {
  const explanation = STAGE_EXPLANATIONS[stage.stage_id] || {
    icon: Info,
    description: stage.description,
    tooltip: stage.description,
  };
  const StageIcon = explanation.icon;

  let StatusIcon = CheckCircle2;
  let statusBadgeClass = "text-emerald-700 bg-emerald-50 border-emerald-200";
  let statusLabel = "Success";

  if (stage.status === "failed") {
    StatusIcon = XCircle;
    statusBadgeClass = "text-rose-700 bg-rose-50 border-rose-200";
    statusLabel = "Failed";
  } else if (stage.status === "fallback") {
    StatusIcon = AlertTriangle;
    statusBadgeClass = "text-amber-800 bg-amber-50 border-amber-300";
    statusLabel = "Fallback";
  } else if (stage.status === "skipped") {
    StatusIcon = Info;
    statusBadgeClass = "text-text-tertiary bg-surface-container border-border-subtle";
    statusLabel = "Skipped";
  }

  // Sanitize details: strictly filter out any sensitive keys
  const safeDetails = Object.entries(stage.details || {}).filter(
    ([key]) => !SENSITIVE_KEYS.has(key.toLowerCase())
  );

  return (
    <div className="border border-border-subtle/70 bg-surface-elevated/40 hover:bg-surface-elevated rounded-md transition-colors overflow-hidden">
      <div
        className="flex items-start gap-2.5 p-2 cursor-pointer select-none"
        onClick={onToggle}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onToggle();
          }
        }}
        aria-expanded={isExpanded}
        aria-label={`${stage.stage_name} - ${statusLabel}`}
      >
        <div className="mt-0.5 text-text-tertiary">
          <StageIcon className="w-3.5 h-3.5 text-brand" />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="font-mono text-[10px] text-text-tertiary">
                {String(stage.order).padStart(2, "0")}
              </span>
              <span className="font-medium text-text-primary text-xs">
                {stage.stage_name}
              </span>
              <span
                className={`inline-flex items-center gap-1 text-[9px] font-medium px-1.5 py-0.2 rounded border ${statusBadgeClass}`}
              >
                <StatusIcon className="w-2.5 h-2.5 shrink-0" />
                <span>{statusLabel}</span>
              </span>
            </div>

            <span className="text-[10px] text-text-tertiary font-mono shrink-0">
              {(stage.duration_ms || 0).toFixed(0)} ms
            </span>
          </div>

          <p className="text-[11px] text-text-secondary truncate mt-0.5" title={explanation.description}>
            {explanation.description}
          </p>

          {stage.fallback_used && stage.fallback_reason && (
            <div className="flex items-center gap-1 text-[10px] text-amber-700 bg-amber-50/60 rounded px-1.5 py-0.5 mt-1 border border-amber-200/60">
              <AlertTriangle className="w-2.5 h-2.5 shrink-0 text-amber-600" />
              <span>{stage.fallback_reason}</span>
            </div>
          )}

          {stage.status === "failed" && stage.error_message && (
            <div className="flex items-center gap-1 text-[10px] text-rose-700 bg-rose-50/60 rounded px-1.5 py-0.5 mt-1 border border-rose-200/60">
              <XCircle className="w-2.5 h-2.5 shrink-0 text-rose-600" />
              <span>{stage.error_message}</span>
            </div>
          )}
        </div>

        <div className="mt-0.5 text-text-tertiary shrink-0">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </div>
      </div>
      
      {isExpanded && (
        <div className="px-3 pb-3 pt-2 border-t border-border-subtle/50 bg-surface-low/60 space-y-2.5">
          <div className="text-[11px] text-text-tertiary italic bg-surface-elevated/70 p-2 rounded border border-border-subtle">
            <span className="font-semibold text-text-secondary not-italic mr-1">
              About this stage:
            </span>
            {explanation.tooltip}
          </div>

          {(stage.input_count !== undefined && stage.input_count !== null) ||
          (stage.output_count !== undefined && stage.output_count !== null) ? (
            <div className="flex items-center gap-3 text-[11px] text-text-secondary font-mono">
              {stage.input_count !== undefined && stage.input_count !== null && (
                <span>
                  <span className="text-text-tertiary">In:</span>{" "}
                  <span className="font-semibold text-text-primary">{stage.input_count}</span>
                </span>
              )}
              {stage.input_count !== undefined &&
                stage.input_count !== null &&
                stage.output_count !== undefined &&
                stage.output_count !== null && (
                  <span className="text-text-tertiary">→</span>
                )}
              {stage.output_count !== undefined && stage.output_count !== null && (
                <span>
                  <span className="text-text-tertiary">Out:</span>{" "}
                  <span className="font-semibold text-text-primary">{stage.output_count}</span>
                </span>
              )}
            </div>
          ) : null}

          {safeDetails.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {safeDetails.map(([key, val]) => (
                <div
                  key={key}
                  className="bg-surface-elevated p-1.5 rounded border border-border-subtle flex flex-col justify-between overflow-hidden"
                >
                  <span className="text-[9px] uppercase tracking-wider text-text-tertiary font-mono truncate" title={key}>
                    {key}
                  </span>
                  <span
                    className="text-[11px] text-text-primary font-medium truncate font-mono mt-0.5"
                    title={typeof val === "object" ? JSON.stringify(val) : String(val ?? "N/A")}
                  >
                    {typeof val === "boolean"
                      ? val
                        ? "Yes"
                        : "No"
                      : typeof val === "object"
                      ? JSON.stringify(val)
                      : String(val ?? "N/A")}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-text-tertiary">
              No additional telemetry parameters for this stage.
            </div>
          )}

          {(stage.stage_id === "evidence_selection" || stage.stage_id === "citation_validation") &&
            citations.length > 0 &&
            onInspectEvidence && (
              <div className="flex justify-end pt-1">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onInspectEvidence(citations[0]);
                  }}
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-brand hover:text-brand-dark bg-brand-light/50 hover:bg-brand-light px-2 py-1 rounded transition-colors"
                >
                  <Maximize2 className="w-3 h-3" />
                  <span>Inspect evidence technical details (Level 4)</span>
                </button>
              </div>
            )}
        </div>
      )}
    </div>
  );
};
