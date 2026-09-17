"use client";

import React, { useState, useEffect } from "react";
import {
  X,
  FileText,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Table as TableIcon,
  Image as ImageIcon,
  Cpu,
} from "lucide-react";
import { Citation } from "@/types";
import { apiClient } from "@/lib/api";
import {
  getCitationLabel,
  getEvidenceType,
  getValidationStatus,
  extractBoundingBox,
} from "@/lib/citationUtils";
import { Button } from "@/components/ui/Button";

export interface EvidenceInspectorProps {
  isOpen: boolean;
  onClose: () => void;
  citations: Citation[];
  initialCitation?: Citation | null;
  documentName?: string;
  isGrounded?: boolean;
  onNavigateToPage: (pageNumber: number, snippet?: string, citation?: Citation) => void;
}

export const EvidenceInspector: React.FC<EvidenceInspectorProps> = ({
  isOpen,
  onClose,
  citations,
  initialCitation,
  documentName,
  isGrounded = true,
  onNavigateToPage,
}) => {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    if (initialCitation && citations.length > 0) {
      const idx = citations.findIndex(
        (c) =>
          (c.chunk_id && c.chunk_id === initialCitation.chunk_id) ||
          (c.rank && c.rank === initialCitation.rank) ||
          c.reference === initialCitation.reference
      );
      if (idx >= 0) setSelectedIndex(idx);
    } else {
      setSelectedIndex(0);
    }
    setImageError(false);
  }, [initialCitation, citations, isOpen]);

  if (!isOpen || citations.length === 0) return null;

  const currentCitation = citations[selectedIndex] || citations[0];
  const citationLabel = getCitationLabel(currentCitation);
  const { type: evidenceType, isVisual } = getEvidenceType(currentCitation);
  const validation = getValidationStatus(currentCitation, isGrounded);
  const bbox = extractBoundingBox(currentCitation);

  const excerpt =
    currentCitation.evidence_text ||
    currentCitation.text ||
    (isVisual ? "Visual page evidence representation" : "No excerpt available.");

  const previewImageUrl =
    currentCitation.image_url ||
    (isVisual && currentCitation.document_id && currentCitation.page_number
      ? apiClient.getPageImageUrl(currentCitation.document_id, currentCitation.page_number)
      : null);

  const meta = currentCitation.metadata || {};

  const handleJumpToPage = () => {
    onNavigateToPage(currentCitation.page_number, excerpt, currentCitation);
    onClose();
  };

  const renderValidationBadge = () => {
    if (validation.status === "verified") {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-800 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
          <span>Verified</span>
        </span>
      );
    }
    if (validation.status === "review") {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-amber-900 bg-amber-50 px-2.5 py-1 rounded-md border border-amber-300">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
          <span>Needs review</span>
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-rose-800 bg-rose-50 px-2.5 py-1 rounded-md border border-rose-200">
        <XCircle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
        <span>Not verified</span>
      </span>
    );
  };

  const ocrConfidence =
    typeof meta.ocr_confidence === "number"
      ? `${(meta.ocr_confidence * 100).toFixed(1)}%`
      : typeof meta.confidence === "number"
      ? `${(meta.confidence * 100).toFixed(1)}%`
      : null;

  const retrievalChannels =
    Array.isArray(currentCitation.sources) && currentCitation.sources.length > 0
      ? currentCitation.sources.join(" + ")
      : currentCitation.retrieval_type || "Dense Vector";

  const modality =
    typeof meta.modality === "string"
      ? meta.modality
      : isVisual
      ? "Visual / Image"
      : "Text / Semantic";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="bg-surface-elevated border border-border-subtle rounded-xl shadow-xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
        role="dialog"
        aria-modal="true"
        aria-labelledby="inspector-title"
      >
        {/* Header */}
        <div className="p-4 sm:px-6 border-b border-border-subtle flex items-center justify-between gap-4 bg-surface-elevated shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h3 id="inspector-title" className="font-heading text-base font-bold text-text-primary">
                Evidence Inspector
              </h3>
              <span className="text-xs font-mono text-text-tertiary bg-surface-low px-2 py-0.5 rounded border border-border-subtle">
                {selectedIndex + 1} of {citations.length}
              </span>
            </div>
            <p className="text-xs text-text-secondary truncate mt-0.5">
              {documentName || currentCitation.document_id} · {citationLabel}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {renderValidationBadge()}
            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-surface-low transition-colors"
              aria-label="Close inspector"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Citation Selector Tabs if multiple */}
        {citations.length > 1 && (
          <div className="flex items-center gap-1.5 px-4 sm:px-6 py-2 bg-surface-low border-b border-border-subtle overflow-x-auto shrink-0">
            {citations.map((c, i) => {
              const label = getCitationLabel(c);
              const isCurr = i === selectedIndex;
              return (
                <button
                  key={`cit_tab_${i}`}
                  type="button"
                  onClick={() => {
                    setSelectedIndex(i);
                    setImageError(false);
                  }}
                  className={`px-2.5 py-1 rounded text-xs font-medium shrink-0 transition-colors ${
                    isCurr
                      ? "bg-brand text-white shadow-xs"
                      : "bg-surface-elevated text-text-secondary border border-border-subtle hover:bg-surface-container"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>
        )}

        {/* Modal Body */}
        <div className="p-4 sm:p-6 overflow-y-auto space-y-4 flex-1">
          {/* Simple Evidence Info */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 bg-surface-low p-3.5 rounded-lg border border-border-subtle text-xs">
            <div>
              <span className="text-text-tertiary block text-[11px] font-medium">Source Document</span>
              <span className="font-medium text-text-primary truncate block" title={documentName || currentCitation.document_id}>
                {documentName || currentCitation.document_id}
              </span>
            </div>
            <div>
              <span className="text-text-tertiary block text-[11px] font-medium">Location</span>
              <span className="font-semibold text-text-primary">
                Page {currentCitation.page_number}
              </span>
            </div>
            <div>
              <span className="text-text-tertiary block text-[11px] font-medium">Evidence Type</span>
              <span className="font-medium text-text-primary">{evidenceType}</span>
            </div>
          </div>

          {/* Supporting Excerpt */}
          <div className="space-y-1.5">
            <span className="text-xs font-semibold text-text-secondary">Supporting Excerpt</span>
            <blockquote className="text-xs sm:text-sm text-text-primary leading-relaxed bg-surface-elevated p-3.5 rounded-lg border-l-4 border-brand border border-border-subtle italic">
              &ldquo;{excerpt}&rdquo;
            </blockquote>
          </div>

          {/* Visual Evidence Preview (if visual) */}
          {isVisual && previewImageUrl && !imageError && (
            <div className="space-y-1.5">
              <span className="text-xs font-semibold text-text-secondary">Visual Evidence Preview</span>
              <div className="relative rounded-lg border border-border-subtle bg-slate-900/5 p-2 flex items-center justify-center max-h-64 overflow-hidden">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewImageUrl}
                  alt={`Rendered evidence on page ${currentCitation.page_number}`}
                  onError={() => setImageError(true)}
                  className="max-h-60 w-auto object-contain rounded"
                />
                {bbox && (
                  <div
                    className="absolute border-2 border-brand bg-brand/15 rounded-xs"
                    style={{
                      left: bbox.isNormalized ? `${bbox.x * 100}%` : `${bbox.x}px`,
                      top: bbox.isNormalized ? `${bbox.y * 100}%` : `${bbox.y}px`,
                      width: bbox.isNormalized ? `${bbox.width * 100}%` : `${bbox.width}px`,
                      height: bbox.isNormalized ? `${bbox.height * 100}%` : `${bbox.height}px`,
                    }}
                  />
                )}
              </div>
            </div>
          )}

          {/* Collapsible Technical Details (Advanced) */}
          <div className="border border-border-subtle rounded-lg overflow-hidden bg-surface-low/30">
            <button
              type="button"
              onClick={() => setShowTechnicalDetails((p) => !p)}
              className="w-full px-4 py-2.5 flex items-center justify-between text-xs font-semibold text-text-primary hover:bg-surface-low transition-colors text-left"
            >
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-text-tertiary" />
                <span>Technical Details (Level 4 Inspection)</span>
              </div>
              {showTechnicalDetails ? (
                <ChevronDown className="w-4 h-4 text-text-tertiary" />
              ) : (
                <ChevronRight className="w-4 h-4 text-text-tertiary" />
              )}
            </button>

            {showTechnicalDetails && (
              <div className="p-4 border-t border-border-subtle bg-surface-elevated space-y-3 text-xs">
                <div className="text-[11px] text-text-tertiary italic bg-surface-base p-2 rounded border border-border-subtle">
                  <span className="font-semibold text-text-secondary not-italic mr-1">
                    Level 4 Inspection:
                  </span>
                  Detailed retrieval provenance and metadata extracted for this evidence chunk.
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <span className="text-[11px] text-text-tertiary block font-mono">Relevance Score</span>
                    <span className="font-mono text-text-primary font-medium">
                      {typeof currentCitation.score === "number"
                        ? currentCitation.score.toFixed(4)
                        : "N/A"}
                    </span>
                  </div>
                  <div>
                    <span className="text-[11px] text-text-tertiary block font-mono">Retrieval Channel</span>
                    <span className="font-medium text-text-primary">{retrievalChannels}</span>
                  </div>
                  <div>
                    <span className="text-[11px] text-text-tertiary block font-mono">Modality</span>
                    <span className="font-medium text-text-primary">{modality}</span>
                  </div>
                  {ocrConfidence && (
                    <div>
                      <span className="text-[11px] text-text-tertiary block font-mono">OCR Confidence</span>
                      <span className="font-mono text-text-primary">{ocrConfidence}</span>
                    </div>
                  )}
                  {currentCitation.chunk_id && (
                    <div>
                      <span className="text-[11px] text-text-tertiary block font-mono">Chunk ID</span>
                      <span className="font-mono text-[11px] text-text-secondary truncate block" title={currentCitation.chunk_id}>
                        {currentCitation.chunk_id}
                      </span>
                    </div>
                  )}
                  {typeof currentCitation.chunk_index === "number" && (
                    <div>
                      <span className="text-[11px] text-text-tertiary block font-mono">Chunk Index</span>
                      <span className="font-mono text-text-primary">{currentCitation.chunk_index}</span>
                    </div>
                  )}
                </div>

                {bbox && (
                  <div>
                    <span className="text-[11px] text-text-tertiary block font-mono mb-1">Bounding Box</span>
                    <span className="font-mono text-[11px] text-text-secondary bg-surface-low px-2 py-1 rounded border border-border-subtle block">
                      [{bbox.x.toFixed(3)}, {bbox.y.toFixed(3)}, {bbox.width.toFixed(3)}, {bbox.height.toFixed(3)}]
                    </span>
                  </div>
                )}

                <div>
                  <span className="text-[11px] text-text-tertiary block font-mono mb-1">Validation Status</span>
                  <p className="text-text-secondary">{validation.description}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 sm:px-6 border-t border-border-subtle bg-surface-low/50 flex items-center justify-between gap-3 shrink-0">
          <Button variant="outline" size="sm" onClick={onClose}>
            Close
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleJumpToPage}
            leftIcon={<ExternalLink className="w-3.5 h-3.5" />}
          >
            Open in Document Viewer (Page {currentCitation.page_number})
          </Button>
        </div>
      </div>
    </div>
  );
};
