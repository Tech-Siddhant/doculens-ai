"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  PanelLeftClose,
  PanelLeftOpen,
  FileText,
  AlertCircle,
  RefreshCw,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Info,
  Table as TableIcon,
  Image as ImageIcon,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { apiClient } from "@/lib/api";
import { DocumentStatus, Citation } from "@/types";
import {
  getCitationLabel,
  getEvidenceType,
  getValidationStatus,
  extractBoundingBox,
} from "@/lib/citationUtils";

interface DocumentViewerProps {
  documentId: string;
  documentName: string;
  totalPages: number;
  currentPage: number;
  onPageChange: (page: number) => void;
  status?: DocumentStatus;
  activeCitationPage?: number | null;
  activeCitationSnippet?: string | null;
  activeCitation?: Citation | null;
  onInspectCitation?: (citation: Citation) => void;
}

const ZOOM_STEPS = [0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0];
const DEFAULT_ZOOM_INDEX = 2; // 1.0 (100%)

export const DocumentViewer: React.FC<DocumentViewerProps> = ({
  documentId,
  documentName,
  totalPages,
  currentPage,
  onPageChange,
  status = "ready",
  activeCitationPage,
  activeCitationSnippet,
  activeCitation,
  onInspectCitation,
}) => {
  const [zoomIndex, setZoomIndex] = useState(DEFAULT_ZOOM_INDEX);
  const [showThumbnails, setShowThumbnails] = useState(true);
  const [isImageLoading, setIsImageLoading] = useState(true);
  const [imageError, setImageError] = useState(false);
  const [pageInputVal, setPageInputVal] = useState(String(currentPage));

  const containerRef = useRef<HTMLDivElement>(null);
  const zoomScale = ZOOM_STEPS[zoomIndex];
  const maxPages = Math.max(1, totalPages);

  useEffect(() => {
    setPageInputVal(String(currentPage));
    setIsImageLoading(true);
    setImageError(false);
  }, [currentPage, documentId]);

  const handleZoomIn = () => setZoomIndex((p) => Math.min(p + 1, ZOOM_STEPS.length - 1));
  const handleZoomOut = () => setZoomIndex((p) => Math.max(p - 1, 0));
  const handleResetZoom = () => setZoomIndex(DEFAULT_ZOOM_INDEX);

  const goToPage = useCallback(
    (page: number) => {
      const clamped = Math.max(1, Math.min(page, maxPages));
      if (clamped !== currentPage) onPageChange(clamped);
    },
    [currentPage, maxPages, onPageChange]
  );

  const handlePageInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const num = parseInt(pageInputVal, 10);
    if (!isNaN(num)) goToPage(num);
    else setPageInputVal(String(currentPage));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    if (e.key === "ArrowLeft" || e.key === "PageUp") {
      e.preventDefault();
      if (currentPage > 1) goToPage(currentPage - 1);
    } else if (e.key === "ArrowRight" || e.key === "PageDown") {
      e.preventDefault();
      if (currentPage < maxPages) goToPage(currentPage + 1);
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      handleZoomIn();
    } else if (e.key === "-") {
      e.preventDefault();
      handleZoomOut();
    } else if (e.key === "0") {
      e.preventDefault();
      handleResetZoom();
    }
  };

  const currentImageUrl = apiClient.getPageImageUrl(documentId, currentPage);
  const isCitationActiveOnThisPage =
    (activeCitation && activeCitation.page_number === currentPage) ||
    activeCitationPage === currentPage;

  const currentCitationOnPage =
    activeCitation && activeCitation.page_number === currentPage ? activeCitation : null;

  const activeBBox = currentCitationOnPage ? extractBoundingBox(currentCitationOnPage) : null;
  const activeValidation = currentCitationOnPage ? getValidationStatus(currentCitationOnPage) : null;
  const activeEvidenceType = currentCitationOnPage ? getEvidenceType(currentCitationOnPage) : null;

  return (
    <div
      ref={containerRef}
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className="flex flex-col h-full bg-surface-elevated border border-border-subtle rounded-lg overflow-hidden shadow-xs focus:outline-none"
      aria-label="Document Page Viewer"
    >
      {/* 1. Top Controls Bar */}
      <div className="px-4 py-2.5 bg-surface-elevated border-b border-border-subtle flex flex-wrap items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowThumbnails((p) => !p)}
            className="text-text-secondary hover:text-text-primary px-2"
            title={showThumbnails ? "Hide thumbnail drawer" : "Show thumbnail drawer"}
            aria-label={showThumbnails ? "Hide thumbnail drawer" : "Show thumbnail drawer"}
          >
            {showThumbnails ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <PanelLeftOpen className="w-4 h-4" />
            )}
          </Button>

          <div className="h-4 w-px bg-border-subtle mx-1" />

          {/* Page Navigation */}
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              disabled={currentPage <= 1}
              onClick={() => goToPage(currentPage - 1)}
              className="px-2"
              title="Previous page (Left Arrow)"
              aria-label="Previous page"
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>

            <form onSubmit={handlePageInputSubmit} className="flex items-center gap-1 text-xs">
              <input
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                value={pageInputVal}
                onChange={(e) => setPageInputVal(e.target.value)}
                onBlur={() => setPageInputVal(String(currentPage))}
                className="w-10 h-7 text-center rounded border border-border-subtle bg-surface-low text-xs font-mono font-medium focus:border-brand focus:outline-none"
                aria-label="Current page number"
              />
              <span className="text-text-tertiary font-mono">/ {maxPages}</span>
            </form>

            <Button
              variant="ghost"
              size="sm"
              disabled={currentPage >= maxPages}
              onClick={() => goToPage(currentPage + 1)}
              className="px-2"
              title="Next page (Right Arrow)"
              aria-label="Next page"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Zoom & View Controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            disabled={zoomIndex <= 0}
            onClick={handleZoomOut}
            className="px-2"
            title="Zoom out (-)"
            aria-label="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </Button>

          <button
            type="button"
            onClick={handleResetZoom}
            className="px-2 py-1 text-xs font-mono font-medium text-text-secondary hover:text-text-primary hover:bg-surface-low rounded transition-colors"
            title="Reset zoom to 100% (0)"
          >
            {Math.round(zoomScale * 100)}%
          </button>

          <Button
            variant="ghost"
            size="sm"
            disabled={zoomIndex >= ZOOM_STEPS.length - 1}
            onClick={handleZoomIn}
            className="px-2"
            title="Zoom in (+)"
            aria-label="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* 2. Main Workspace (Thumbnails Drawer + Canvas Area) */}
      <div className="flex-1 flex min-h-0 relative overflow-hidden bg-surface-low/50">
        {/* Left Thumbnail Drawer */}
        {showThumbnails && (
          <div className="w-44 border-r border-border-subtle bg-surface-elevated flex flex-col shrink-0 overflow-hidden">
            <div className="p-2 border-b border-border-subtle flex items-center justify-between text-[11px] font-semibold text-text-secondary uppercase tracking-wider">
              <span>Pages ({maxPages})</span>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {Array.from({ length: maxPages }, (_, i) => i + 1).map((pageNum) => {
                const isSelected = pageNum === currentPage;
                const isCited =
                  (activeCitation && activeCitation.page_number === pageNum) ||
                  activeCitationPage === pageNum;
                const thumbUrl = apiClient.getPageImageUrl(documentId, pageNum);

                return (
                  <button
                    key={pageNum}
                    type="button"
                    onClick={() => goToPage(pageNum)}
                    className={`w-full group relative rounded border text-left p-1 transition-all ${
                      isSelected
                        ? "border-brand ring-2 ring-brand/30 bg-brand-light/20"
                        : "border-border-subtle bg-surface-low hover:border-border-strong hover:bg-surface-elevated"
                    }`}
                  >
                    <div className="aspect-[3/4] bg-white rounded overflow-hidden flex items-center justify-center border border-border-subtle/50 relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbUrl}
                        alt={`Page ${pageNum} thumbnail`}
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                      {isCited && (
                        <div className="absolute top-1 right-1 bg-brand text-white p-0.5 rounded shadow-xs">
                          <Sparkles className="w-2.5 h-2.5" />
                        </div>
                      )}
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[10px]">
                      <span
                        className={`font-mono font-medium ${
                          isSelected ? "text-brand" : "text-text-secondary"
                        }`}
                      >
                        p. {pageNum}
                      </span>
                      {isCited && (
                        <span className="text-[9px] font-medium text-brand bg-brand-light px-1 rounded">
                          Cited
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Center Page Canvas */}
        <div className="flex-1 overflow-auto flex flex-col items-center p-4 sm:p-6 min-h-0">
          <div
            className="transition-transform duration-150 origin-top flex flex-col items-center max-w-full"
            style={{ transform: `scale(${zoomScale})` }}
          >
            <div className="bg-surface-elevated border border-border-subtle rounded-md shadow-md overflow-hidden relative min-w-[320px] sm:min-w-[500px] max-w-[800px] select-text">
              {/* Active Grounding Evidence Banner */}
              {isCitationActiveOnThisPage && (
                <div
                  className="bg-brand text-white px-3.5 py-2 text-xs flex items-center justify-between gap-2 shadow-xs"
                  role="status"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <Sparkles className="w-3.5 h-3.5 text-amber-300 shrink-0" />
                    <span className="font-semibold">
                      Active Grounding Source · Page {currentPage}
                    </span>
                    {activeEvidenceType && (
                      <span className="text-[10px] font-medium bg-white/20 text-white px-1.5 py-0.5 rounded">
                        {activeEvidenceType.type}
                      </span>
                    )}
                    {activeValidation && (
                      <span className="text-[10px] font-medium bg-emerald-500/30 text-emerald-100 border border-emerald-400/40 px-1.5 py-0.5 rounded hidden sm:inline-flex items-center gap-1">
                        <CheckCircle2 className="w-2.5 h-2.5" />
                        <span>{activeValidation.label}</span>
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    {(currentCitationOnPage?.evidence_text || activeCitationSnippet) && (
                      <span className="text-[10px] text-blue-100 hidden md:inline truncate max-w-xs italic">
                        &ldquo;
                        {(
                          currentCitationOnPage?.evidence_text ||
                          activeCitationSnippet ||
                          ""
                        ).slice(0, 45)}
                        ...&rdquo;
                      </span>
                    )}
                    {currentCitationOnPage && onInspectCitation && (
                      <button
                        type="button"
                        onClick={() => onInspectCitation(currentCitationOnPage)}
                        className="inline-flex items-center gap-1 text-[10px] bg-white text-brand px-1.5 py-0.5 rounded font-medium hover:bg-blue-50 transition-colors"
                      >
                        <Info className="w-2.5 h-2.5" />
                        <span>Inspect</span>
                      </button>
                    )}
                  </div>
                </div>
              )}

              {/* Page Image & Bounding Box Overlay */}
              <div className="relative min-h-[400px] sm:min-h-[600px] flex items-center justify-center bg-white">
                {isImageLoading && !imageError && (
                  <div
                    className="absolute inset-0 flex flex-col items-center justify-center bg-surface-low/80 z-10 p-4"
                    aria-live="polite"
                  >
                    <RefreshCw className="w-6 h-6 text-brand animate-spin mb-2" />
                    <p className="text-xs font-medium text-text-secondary">
                      Rendering page {currentPage}...
                    </p>
                  </div>
                )}

                {imageError ? (
                  <div
                    className="p-8 flex flex-col items-center text-center max-w-sm"
                    role="alert"
                  >
                    <AlertCircle className="w-8 h-8 text-status-failed mb-2" />
                    <h4 className="font-heading text-sm font-semibold text-text-primary mb-1">
                      Unable to render page {currentPage}
                    </h4>
                    <p className="text-xs text-text-secondary mb-4">
                      The page preview could not be loaded from the backend.
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setImageError(false);
                        setIsImageLoading(true);
                      }}
                      leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
                    >
                      Retry page
                    </Button>
                  </div>
                ) : (
                  <div className="relative w-full h-auto">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      key={`${documentId}_p${currentPage}`}
                      src={currentImageUrl}
                      alt={`Rendered preview of ${documentName}, page ${currentPage}`}
                      onLoad={() => {
                        setIsImageLoading(false);
                        setImageError(false);
                      }}
                      onError={() => {
                        setIsImageLoading(false);
                        setImageError(true);
                      }}
                      className={`w-full h-auto object-contain transition-opacity duration-200 ${
                        isImageLoading ? "opacity-0" : "opacity-100"
                      }`}
                      loading="eager"
                    />

                    {/* Bounding Box Highlight Overlay */}
                    {!isImageLoading && !imageError && activeBBox && (
                      <div
                        className="absolute border-2 border-brand bg-brand/15 shadow-sm rounded-xs pointer-events-none transition-all duration-200 animate-pulse"
                        style={{
                          left: activeBBox.isNormalized
                            ? `${activeBBox.x * 100}%`
                            : `${activeBBox.x}px`,
                          top: activeBBox.isNormalized
                            ? `${activeBBox.y * 100}%`
                            : `${activeBBox.y}px`,
                          width: activeBBox.isNormalized
                            ? `${activeBBox.width * 100}%`
                            : `${activeBBox.width}px`,
                          height: activeBBox.isNormalized
                            ? `${activeBBox.height * 100}%`
                            : `${activeBBox.height}px`,
                        }}
                      >
                        {activeEvidenceType && (
                          <span className="absolute -top-5 left-0 bg-brand text-white text-[9px] font-medium px-1.5 py-0.5 rounded shadow">
                            {activeEvidenceType.type}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="px-4 py-2 bg-surface-low/40 border-t border-border-subtle flex items-center justify-between text-[10px] text-text-tertiary font-mono">
                <span>
                  Page {currentPage} of {maxPages}
                </span>
                <span>DocuLens Document Viewer</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
