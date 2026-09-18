"use client";

import React from "react";
import {
  Terminal,
  Database,
  Cpu,
  ShieldCheck,
  Sparkles,
  Layers,
  FileText,
  Clock,
  CheckCircle2,
  Info,
} from "lucide-react";
import { PipelineTrace, Citation, DocumentItem } from "@/types";
import { PipelineTimeline } from "./PipelineTimeline";

interface TechnicalProcessSidePanelProps {
  document: DocumentItem;
  trace?: PipelineTrace | null;
  citations?: Citation[];
  onNavigateToPage?: (pageNumber: number, snippet?: string, citation?: Citation) => void;
  onInspectEvidence?: (citation?: Citation) => void;
  onSwitchToPdf?: () => void;
}

export const TechnicalProcessSidePanel: React.FC<TechnicalProcessSidePanelProps> = ({
  document,
  trace,
  citations = [],
  onNavigateToPage,
  onInspectEvidence,
  onSwitchToPdf,
}) => {
  return (
    <div className="flex flex-col h-full bg-surface-elevated border border-border-subtle rounded-lg overflow-hidden shadow-xs">
      {/* Header */}
      <div className="px-4 py-3 bg-surface-elevated border-b border-border-subtle flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded bg-brand text-white flex items-center justify-center font-mono">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <h2 className="font-heading text-xs font-bold text-text-primary uppercase tracking-wider font-mono flex items-center gap-2">
              <span>Technical Process</span>
              <span className="text-[10px] px-1.5 py-0.2 bg-emerald-50 text-status-ready border border-emerald-200 rounded font-mono uppercase">
                Observable
              </span>
            </h2>
            <p className="text-[11px] text-text-tertiary">
              Retrieval pipeline telemetry &amp; LLM reasoning trace
            </p>
          </div>
        </div>

        {trace ? (
          <div className="flex items-center gap-2 font-mono text-[11px] text-text-secondary bg-surface-low px-2 py-1 rounded border border-border-subtle">
            <Clock className="w-3 h-3 text-brand" />
            <span>{Math.round(trace.summary.total_duration_ms)}ms</span>
            <span>·</span>
            <span className="text-status-ready">{trace.stages.length} stages</span>
          </div>
        ) : (
          <span className="font-mono text-[11px] text-text-tertiary bg-surface-low px-2 py-0.5 rounded border border-border-subtle">
            Ready
          </span>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0 bg-surface-low/30">
        {/* Real-time Query Pipeline Trace (If query executed) */}
        {trace ? (
          <div className="space-y-4">
            <div className="bg-surface-elevated border border-border-subtle rounded-lg p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-text-tertiary font-medium">QUERY EXECUTION TRACE</span>
                <span className="text-brand font-semibold">{trace.pipeline_id}</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-[11px]">
                <div className="p-2 bg-surface-low rounded border border-border-subtle">
                  <span className="text-text-tertiary block text-[10px]">RETRIEVAL</span>
                  <span className="font-semibold text-text-primary">
                    {trace.summary.retrieval_duration_ms
                      ? `${Math.round(trace.summary.retrieval_duration_ms)}ms`
                      : "—"}
                  </span>
                </div>
                <div className="p-2 bg-surface-low rounded border border-border-subtle">
                  <span className="text-text-tertiary block text-[10px]">RERANKING</span>
                  <span className="font-semibold text-text-primary">
                    {trace.summary.reranking_duration_ms
                      ? `${Math.round(trace.summary.reranking_duration_ms)}ms`
                      : "—"}
                  </span>
                </div>
                <div className="p-2 bg-surface-low rounded border border-border-subtle">
                  <span className="text-text-tertiary block text-[10px]">GENERATION</span>
                  <span className="font-semibold text-text-primary">
                    {trace.summary.generation_duration_ms
                      ? `${Math.round(trace.summary.generation_duration_ms)}ms`
                      : "—"}
                  </span>
                </div>
                <div className="p-2 bg-surface-low rounded border border-border-subtle">
                  <span className="text-text-tertiary block text-[10px]">VALIDATION</span>
                  <span className="font-semibold text-text-primary">
                    {trace.summary.citation_validation_duration_ms
                      ? `${Math.round(trace.summary.citation_validation_duration_ms)}ms`
                      : "—"}
                  </span>
                </div>
              </div>
            </div>

            {/* Complete Interactive Stages Breakdown */}
            <div className="space-y-2">
              <h3 className="text-xs font-heading font-semibold text-text-secondary uppercase tracking-wider font-mono">
                Pipeline Stages Breakdown
              </h3>
              <PipelineTimeline
                trace={trace}
                citations={citations}
                onNavigateToPage={onNavigateToPage}
                onInspectEvidence={onInspectEvidence}
              />
            </div>
          </div>
        ) : (
          /* Awaiting Query Board */
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-surface-elevated border border-border-subtle text-center space-y-2">
              <div className="w-10 h-10 rounded-full bg-brand-light text-brand flex items-center justify-center mx-auto">
                <Terminal className="w-5 h-5" />
              </div>
              <h3 className="font-heading text-sm font-semibold text-text-primary">
                Live Pipeline Telemetry Ready
              </h3>
              <p className="text-xs text-text-secondary max-w-sm mx-auto leading-relaxed">
                Ask any question in the assistant panel on the left to see real-time hybrid retrieval scores, RRF fusion, cross-encoder reranking, and citation verification here.
              </p>
            </div>

            {/* Document Indexing Baseline */}
            <div className="bg-surface-elevated border border-border-subtle rounded-lg p-4 space-y-3">
              <div className="flex items-center gap-2 border-b border-border-subtle pb-2">
                <Database className="w-4 h-4 text-brand" />
                <h4 className="font-heading text-xs font-semibold text-text-primary uppercase tracking-wider font-mono">
                  Document Indexing Specifications
                </h4>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 bg-surface-low rounded border border-border-subtle space-y-1">
                  <span className="text-[10px] text-text-tertiary uppercase">Vector Store</span>
                  <p className="font-semibold text-text-primary">Dense Vector Index</p>
                  <p className="text-[10px] text-text-secondary">Semantic dense embeddings per chunk</p>
                </div>
                <div className="p-2.5 bg-surface-low rounded border border-border-subtle space-y-1">
                  <span className="text-[10px] text-text-tertiary uppercase">Keyword Retriever</span>
                  <p className="font-semibold text-text-primary">BM25 Lexical Index</p>
                  <p className="text-[10px] text-text-secondary">Inverted index tokenized per document</p>
                </div>
                <div className="p-2.5 bg-surface-low rounded border border-border-subtle space-y-1">
                  <span className="text-[10px] text-text-tertiary uppercase">Generation Engine</span>
                  <p className="font-semibold text-text-primary">Grounded LLM Generator</p>
                  <p className="text-[10px] text-text-secondary">Strict citation validation &amp; verification</p>
                </div>
                <div className="p-2.5 bg-surface-low rounded border border-border-subtle space-y-1">
                  <span className="text-[10px] text-text-tertiary uppercase">Document Metadata</span>
                  <p className="font-semibold text-text-primary">{document.pageCount} Pages Extracted</p>
                  <p className="text-[10px] text-text-secondary">Status: {document.status.toUpperCase()}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Global Architecture Specifications */}
        <div className="p-3 bg-surface-elevated border border-border-subtle rounded-lg text-xs font-mono space-y-1.5 text-text-tertiary">
          <div className="flex items-center justify-between">
            <span>FUSION METHOD:</span>
            <span className="text-text-primary font-semibold">Reciprocal Rank Fusion (k=60)</span>
          </div>
          <div className="flex items-center justify-between">
            <span>ISOLATION:</span>
            <span className="text-status-ready font-semibold">Strict Document ID Filter</span>
          </div>
          <div className="flex items-center justify-between">
            <span>GROUNDING:</span>
            <span className="text-text-primary font-semibold">N-Gram &amp; Levenshtein Verification</span>
          </div>
        </div>
      </div>
    </div>
  );
};
