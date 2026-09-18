"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ChevronLeft,
  FileText,
  AlertCircle,
  RefreshCw,
  ArrowLeft,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { EmptyState } from "@/components/ui/EmptyState";
import { DocumentViewer } from "@/components/viewer/DocumentViewer";
import { QuestionAnswerPanel } from "@/components/qa/QuestionAnswerPanel";
import { TechnicalProcessSidePanel } from "@/components/qa/TechnicalProcessSidePanel";
import { apiClient, mapBackendDocToItem } from "@/lib/api";
import { DocumentItem, Citation } from "@/types";

export default function DocumentWorkspacePage() {
  const params = useParams();
  const rawId = params?.id;
  const documentId = Array.isArray(rawId) ? rawId[0] : (rawId as string);

  const [document, setDocument] = useState<DocumentItem | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [currentPage, setCurrentPage] = useState<number>(1);
  const [activeCitationPage, setActiveCitationPage] = useState<number | null>(null);
  const [activeCitationSnippet, setActiveCitationSnippet] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);

  const fetchDocument = useCallback(async () => {
    if (!documentId) return;
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiClient.getDocument(documentId);
      const mapped = mapBackendDocToItem(res);
      setDocument(mapped);
    } catch (err: unknown) {
      const msg =
        (err as { message?: string })?.message ||
        "The requested document could not be found or loaded.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    fetchDocument();
  }, [fetchDocument]);

  const [sideView, setSideView] = useState<"pdf" | "process">("pdf");
  const [latestTrace, setLatestTrace] = useState<any>(null);
  const [latestCitations, setLatestCitations] = useState<Citation[]>([]);

  const handleNavigateToPage = useCallback(
    (pageNumber: number, snippet?: string, citation?: Citation) => {
      setCurrentPage(pageNumber);
      setActiveCitationPage(pageNumber);
      if (snippet !== undefined) setActiveCitationSnippet(snippet);
      if (citation !== undefined) setActiveCitation(citation);
    },
    []
  );

  const handlePageChange = useCallback((pageNumber: number) => {
    setCurrentPage(pageNumber);
  }, []);

  if (isLoading) {
    return (
      <div className="h-[calc(100vh-8rem)] flex flex-col items-center justify-center space-y-3" aria-live="polite">
        <RefreshCw className="w-8 h-8 text-brand animate-spin" />
        <p className="font-heading text-sm font-semibold text-text-primary">
          Opening document workspace...
        </p>
        <p className="text-xs text-text-tertiary">
          Loading document structure and page renderer
        </p>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="max-w-2xl mx-auto py-12 px-4">
        <EmptyState
          icon={<AlertCircle className="w-8 h-8 text-status-failed" />}
          title="Document not found"
          description={error || `Could not find document with identifier "${documentId}".`}
          action={
            <div className="flex items-center gap-3">
              <Link href="/documents">
                <Button variant="primary" size="md" leftIcon={<ArrowLeft className="w-4 h-4" />}>
                  Back to Documents
                </Button>
              </Link>
              <Button variant="outline" size="md" onClick={fetchDocument} leftIcon={<RefreshCw className="w-4 h-4" />}>
                Retry
              </Button>
            </div>
          }
        />
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-6.5rem)] flex flex-col space-y-3">
      {/* 1. Top Breadcrumb & Metadata Navigation */}
      <div className="flex items-center justify-between gap-3 shrink-0 pb-1">
        <div className="flex items-center gap-2 text-xs text-text-tertiary min-w-0">
          <Link
            href="/documents"
            className="flex items-center gap-1 hover:text-brand transition-colors font-medium text-text-secondary"
          >
            <ChevronLeft className="w-4 h-4" />
            <span>Documents</span>
          </Link>
          <span>/</span>
          <span className="font-semibold text-text-primary truncate max-w-xs sm:max-w-md">
            {document.name}
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden sm:inline-flex items-center gap-1 text-xs text-text-secondary bg-surface-low px-2.5 py-1 rounded border border-border-subtle font-mono">
            <FileText className="w-3.5 h-3.5 text-text-tertiary" />
            {document.pageCount} {document.pageCount === 1 ? "page" : "pages"}
          </span>
          <StatusBadge status={document.status} size="sm" />
        </div>
      </div>

      {/* 2. Responsive Split Workspace */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0 overflow-y-auto lg:overflow-hidden">
        {/* Left Column: Q&A Interaction Panel (5 cols on LG) */}
        <div className="lg:col-span-5 h-[480px] sm:h-[540px] lg:h-full min-h-0 flex flex-col">
          <QuestionAnswerPanel
            documentId={document.id}
            documentName={document.name}
            onNavigateToPage={handleNavigateToPage}
            activeCitationPage={activeCitationPage}
            activeCitation={activeCitation}
            onSelectCitation={setActiveCitation}
            onTraceUpdate={(trace, citations) => {
              setLatestTrace(trace);
              setLatestCitations(citations);
            }}
            isSideProcessActive={sideView === "process"}
            onToggleSideProcess={() =>
              setSideView((prev) => (prev === "process" ? "pdf" : "process"))
            }
          />
        </div>

        {/* Right Column: Document Viewer or Technical Process (7 cols on LG) */}
        <div className="lg:col-span-7 h-[520px] sm:h-[600px] lg:h-full min-h-0 flex flex-col space-y-2">
          {/* Side View Switcher Tabs */}
          <div className="flex items-center justify-between px-1 shrink-0">
            <div className="inline-flex items-center p-0.5 bg-surface-low rounded-lg border border-border-subtle" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={sideView === "pdf"}
                onClick={() => setSideView("pdf")}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  sideView === "pdf"
                    ? "bg-surface-elevated text-text-primary shadow-xs font-semibold"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Document PDF</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={sideView === "process"}
                onClick={() => setSideView("process")}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-all ${
                  sideView === "process"
                    ? "bg-surface-elevated text-brand shadow-xs font-semibold"
                    : "text-text-secondary hover:text-text-primary"
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>Technical Process</span>
                {latestTrace && (
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                )}
              </button>
            </div>

            <span className="text-[11px] font-mono text-text-tertiary hidden sm:inline">
              {sideView === "pdf" ? "Page Viewer" : "Observable Telemetry"}
            </span>
          </div>

          {/* Tab Content */}
          <div className="flex-1 min-h-0">
            {sideView === "pdf" ? (
              <DocumentViewer
                documentId={document.id}
                documentName={document.name}
                totalPages={document.pageCount}
                currentPage={currentPage}
                onPageChange={handlePageChange}
                status={document.status}
                activeCitationPage={activeCitationPage}
                activeCitationSnippet={activeCitationSnippet}
                activeCitation={activeCitation}
              />
            ) : (
              <TechnicalProcessSidePanel
                document={document}
                trace={latestTrace}
                citations={latestCitations}
                onNavigateToPage={(p: number, s?: string, c?: Citation) => {
                  handleNavigateToPage(p, s, c);
                  setSideView("pdf");
                }}
                onSwitchToPdf={() => setSideView("pdf")}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
