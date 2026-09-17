"use client";

import React, { useState } from "react";
import {
  FileText,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  Info,
  Maximize2,
  Table as TableIcon,
  Image as ImageIcon,
} from "lucide-react";
import { Citation } from "@/types";
import { apiClient } from "@/lib/api";
import {
  getCitationLabel,
  getEvidenceType,
  getValidationStatus,
  extractBoundingBox,
} from "@/lib/citationUtils";

export interface EvidenceCardProps {
  citation: Citation;
  documentName?: string;
  isGrounded?: boolean;
  isSelected?: boolean;
  onNavigateToPage: (pageNumber: number, snippet?: string, citation?: Citation) => void;
  onInspect?: (citation: Citation) => void;
  compact?: boolean;
  className?: string;
}

export const EvidenceCard: React.FC<EvidenceCardProps> = ({
  citation,
  documentName,
  isGrounded = true,
  isSelected = false,
  onNavigateToPage,
  onInspect,
  compact = false,
  className = "",
}) => {
  const [imageError, setImageError] = useState(false);
  const citationLabel = getCitationLabel(citation);
  const { type: evidenceType, isVisual } = getEvidenceType(citation);
  const validation = getValidationStatus(citation, isGrounded);
  const bbox = extractBoundingBox(citation);

  const excerpt =
    citation.evidence_text || citation.text || (isVisual ? "Visual page evidence" : "");

  const previewImageUrl =
    citation.image_url ||
    (isVisual && citation.document_id && citation.page_number
      ? apiClient.getPageImageUrl(citation.document_id, citation.page_number)
      : null);

  const handleCardClick = () => {
    onNavigateToPage(citation.page_number, excerpt, citation);
  };

  const getValidationBadge = () => {
    if (validation.status === "verified") {
      return (
        <span
          className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200"
          title={validation.description}
        >
          <CheckCircle2 className="w-3 h-3 text-emerald-600 shrink-0" />
          <span>Verified</span>
        </span>
      );
    }
    if (validation.status === "review") {
      return (
        <span
          className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-300"
          title={validation.description}
        >
          <AlertTriangle className="w-3 h-3 text-amber-600 shrink-0" />
          <span>Needs review</span>
        </span>
      );
    }
    return (
      <span
        className="inline-flex items-center gap-1 text-[11px] font-medium text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200"
        title={validation.description}
      >
        <XCircle className="w-3 h-3 text-rose-600 shrink-0" />
        <span>Not verified</span>
      </span>
    );
  };

  const typeIcon = evidenceType.toLowerCase().includes("table") ? (
    <TableIcon className="w-3.5 h-3.5 text-blue-600" />
  ) : isVisual ? (
    <ImageIcon className="w-3.5 h-3.5 text-indigo-600" />
  ) : (
    <FileText className="w-3.5 h-3.5 text-text-secondary" />
  );

  return (
    <div
      className={`group relative rounded-lg border transition-all duration-150 overflow-hidden ${
        isSelected
          ? "border-brand bg-brand-light/30 shadow-xs ring-1 ring-brand/40"
          : "border-border-subtle bg-surface-elevated hover:border-border-strong hover:shadow-xs"
      } ${className}`}
    >
      <div className="p-3 space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="inline-flex items-center gap-1 text-xs font-semibold text-text-primary">
                {typeIcon}
                <span>{citationLabel}</span>
              </span>
              <span className="text-[10px] font-medium text-text-tertiary px-1.5 py-0.5 bg-surface-low rounded border border-border-subtle">
                {evidenceType}
              </span>
            </div>
            {documentName && (
              <p className="text-[11px] text-text-tertiary truncate mt-0.5" title={documentName}>
                {documentName}
              </p>
            )}
          </div>
          <div className="shrink-0">{getValidationBadge()}</div>
        </div>

        {isVisual && previewImageUrl && !imageError && (
          <div
            onClick={handleCardClick}
            className="relative cursor-pointer overflow-hidden rounded border border-border-subtle bg-slate-900/5 aspect-[16/9] max-h-32 group/img flex items-center justify-center"
            title="Click to view in document"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewImageUrl}
              alt={`Evidence preview from page ${citation.page_number}`}
              onError={() => setImageError(true)}
              className="w-full h-full object-contain group-hover/img:scale-[1.02] transition-transform duration-200"
              loading="lazy"
            />
            {bbox && (
              <div
                className="absolute border-2 border-brand bg-brand/10 pointer-events-none rounded-xs"
                style={{
                  left: bbox.isNormalized ? `${bbox.x * 100}%` : `${bbox.x}px`,
                  top: bbox.isNormalized ? `${bbox.y * 100}%` : `${bbox.y}px`,
                  width: bbox.isNormalized ? `${bbox.width * 100}%` : `${bbox.width}px`,
                  height: bbox.isNormalized ? `${bbox.height * 100}%` : `${bbox.height}px`,
                }}
              />
            )}
            <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/10 transition-colors flex items-center justify-center opacity-0 group-hover/img:opacity-100">
              <span className="inline-flex items-center gap-1 text-[11px] font-medium text-white bg-black/70 px-2 py-0.5 rounded shadow">
                <Maximize2 className="w-3 h-3" />
                <span>Open evidence</span>
              </span>
            </div>
          </div>
        )}

        {excerpt && !compact && (
          <blockquote className="text-xs text-text-secondary leading-relaxed border-l-2 border-brand/50 pl-2.5 py-0.5 italic bg-surface-low/50 rounded-r line-clamp-3">
            &ldquo;{excerpt}&rdquo;
          </blockquote>
        )}

        <div className="flex items-center justify-between gap-2 pt-1 border-t border-border-subtle/60 text-[11px]">
          <button
            type="button"
            onClick={handleCardClick}
            className="inline-flex items-center gap-1 font-medium text-brand hover:text-brand-dark transition-colors py-0.5 px-1 rounded hover:bg-brand-light"
            aria-label={`Open evidence on page ${citation.page_number}`}
          >
            <ExternalLink className="w-3 h-3" />
            <span>Open evidence (p. {citation.page_number})</span>
          </button>

          {onInspect && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onInspect(citation);
              }}
              className="inline-flex items-center gap-1 text-text-tertiary hover:text-text-primary transition-colors py-0.5 px-1.5 rounded hover:bg-surface-low"
              aria-label="Inspect evidence details"
            >
              <Info className="w-3 h-3" />
              <span>Inspect</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
