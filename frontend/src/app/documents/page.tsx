"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Search, Plus, ArrowUpDown, X, RefreshCw, Layers } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { EmptyState } from "@/components/ui/EmptyState";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { SkeletonTable } from "@/components/ui/LoadingState";
import { UploadModal } from "@/components/documents/UploadModal";
import { DocumentTable } from "@/components/documents/DocumentTable";
import { apiClient, mapBackendDocToItem } from "@/lib/api";
import { DocumentItem } from "@/types";

type SortOption = "newest" | "oldest" | "name_asc" | "size_desc";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortOption>("newest");
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.getDocuments();
      const mapped = response.map(mapBackendDocToItem);
      setDocuments(mapped);
    } catch (err: unknown) {
      const msg =
        (err as { message?: string })?.message ||
        "Failed to load documents. Please check backend connection.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const filteredAndSortedDocuments = useMemo(() => {
    let result = [...documents];

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (doc) =>
          doc.name.toLowerCase().includes(q) ||
          doc.id.toLowerCase().includes(q) ||
          (doc.description && doc.description.toLowerCase().includes(q))
      );
    }

    if (statusFilter !== "all") {
      result = result.filter((doc) => doc.status === statusFilter);
    }

    result.sort((a, b) => {
      if (sortBy === "name_asc") return a.name.localeCompare(b.name);
      if (sortBy === "size_desc") return (b.sizeBytes || 0) - (a.sizeBytes || 0);

      const timeA = a.uploadedAtISO ? new Date(a.uploadedAtISO).getTime() : NaN;
      const timeB = b.uploadedAtISO ? new Date(b.uploadedAtISO).getTime() : NaN;
      if (!Number.isNaN(timeA) && !Number.isNaN(timeB)) {
        return sortBy === "oldest" ? timeA - timeB : timeB - timeA;
      }
      // Fallback for documents without a backend timestamp (identifier ordering)
      return sortBy === "oldest" ? a.id.localeCompare(b.id) : b.id.localeCompare(a.id);
    });

    return result;
  }, [documents, searchQuery, statusFilter, sortBy]);

  const counts = useMemo(() => ({
    all: documents.length,
    ready: documents.filter((d) => d.status === "ready").length,
    processing: documents.filter((d) => d.status === "processing").length,
    needs_attention: documents.filter((d) => d.status === "needs_attention").length,
    failed: documents.filter((d) => d.status === "failed").length,
  }), [documents]);

  const tabs = [
    { label: "All", value: "all", count: counts.all },
    { label: "Ready", value: "ready", count: counts.ready },
    { label: "Processing", value: "processing", count: counts.processing },
    { label: "Needs attention", value: "needs_attention", count: counts.needs_attention },
    { label: "Failed", value: "failed", count: counts.failed },
  ];

  return (
    <div className="space-y-6">
      {/* 1. Header with Title, Subtitle, and CTA */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-border-subtle">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight text-text-primary">
            Documents
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            Upload, manage, and explore multimodal documents with AI assistance.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchDocuments}
            isLoading={isLoading}
            leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
            title="Refresh list"
          >
            Refresh
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={() => setIsUploadOpen(true)}
            leftIcon={<Plus className="w-4 h-4" />}
          >
            Upload document
          </Button>
        </div>
      </div>

      {/* 2. Error State Banner */}
      {error && (
        <ErrorBanner
          title="Unable to load documents"
          message={error}
          onRetry={fetchDocuments}
        />
      )}

      {/* 3. Search and Status Filtering Controls */}
      <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 bg-surface-elevated border border-border-subtle p-3 rounded shadow-xs">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-tertiary pointer-events-none" />
          <input
            type="text"
            placeholder="Search documents by filename or ID..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-8 py-1.5 text-sm bg-surface-low border border-border-subtle rounded text-text-primary placeholder:text-text-tertiary focus:outline-none focus:border-brand focus:ring-1 focus:ring-brand transition-colors"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-tertiary hover:text-text-primary p-0.5"
              aria-label="Clear search"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <div className="flex items-center gap-1 overflow-x-auto pb-1 lg:pb-0 scrollbar-none">
          {tabs.map((tab) => {
            const isActive = statusFilter === tab.value;
            return (
              <button
                key={tab.value}
                type="button"
                onClick={() => setStatusFilter(tab.value)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded transition-colors whitespace-nowrap ${
                  isActive
                    ? "bg-text-primary text-white shadow-xs"
                    : "bg-surface-low text-text-secondary hover:text-text-primary hover:bg-surface-container"
                }`}
              >
                <span>{tab.label}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isActive ? "bg-white/20 text-white" : "bg-surface-container text-text-tertiary"
                  }`}
                >
                  {tab.count}
                </span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <ArrowUpDown className="w-3.5 h-3.5 text-text-tertiary" />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortOption)}
            className="text-xs bg-surface-low border border-border-subtle rounded px-2.5 py-1.5 text-text-secondary focus:outline-none focus:border-brand"
            aria-label="Sort documents"
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="name_asc">Name (A-Z)</option>
            <option value="size_desc">Size (largest)</option>
          </select>
        </div>
      </div>

      {/* 4. Table / Empty / Loading Display */}
      {isLoading ? (
        <SkeletonTable rows={5} />
      ) : documents.length === 0 ? (
        <EmptyState
          icon={<Layers className="w-7 h-7 text-brand" />}
          title="No documents yet"
          description="Upload your first research paper or PDF document to begin extracting multimodal evidence and asking grounded questions."
          action={
            <Button
              variant="primary"
              size="md"
              onClick={() => setIsUploadOpen(true)}
              leftIcon={<Plus className="w-4 h-4" />}
            >
              Upload document
            </Button>
          }
        />
      ) : filteredAndSortedDocuments.length === 0 ? (
        <EmptyState
          icon={<Search className="w-6 h-6 text-text-tertiary" />}
          title="No matching documents"
          description={`No documents match your filter "${statusFilter}" and search query "${searchQuery}".`}
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setSearchQuery("");
                setStatusFilter("all");
              }}
            >
              Reset filters
            </Button>
          }
        />
      ) : (
        <DocumentTable
          documents={filteredAndSortedDocuments}
          totalCount={documents.length}
        />
      )}

      {/* 5. Ingestion Upload Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={fetchDocuments}
      />
    </div>
  );
}
