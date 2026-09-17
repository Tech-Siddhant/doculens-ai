"use client";

import React, { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import {
  UploadCloud,
  FileText,
  ArrowRight,
  Sparkles,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { Card } from "@/components/ui/Card";
import { ErrorBanner } from "@/components/ui/ErrorBanner";
import { EmptyState } from "@/components/ui/EmptyState";
import { SkeletonCard } from "@/components/ui/LoadingState";
import { DocumentItem } from "@/types";
import { UploadModal } from "@/components/documents/UploadModal";
import {
  apiClient,
  mapBackendDocToItem,
  MAX_UPLOAD_SIZE_LABEL,
} from "@/lib/api";

const RECENT_LIMIT = 4;

export default function HomePage() {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [recentDocs, setRecentDocs] = useState<DocumentItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRecentDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiClient.getDocuments();
      setRecentDocs(response.map(mapBackendDocToItem).slice(0, RECENT_LIMIT));
    } catch (err: unknown) {
      const msg =
        (err as { message?: string })?.message ||
        "Failed to load recent documents. Please check backend connection.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRecentDocuments();
  }, [fetchRecentDocuments]);

  return (
    <div className="space-y-8">

      <div className="space-y-2">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-brand-light text-brand border border-blue-100">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Multimodal Document Intelligence</span>
        </div>
        <h1 className="font-heading text-2xl sm:text-3xl font-bold tracking-tight text-text-primary">
          Upload a document, then ask questions about it
        </h1>
        <p className="text-text-secondary text-sm sm:text-base max-w-3xl">
          DocuLens AI retrieves evidence from complex research PDFs to produce grounded answers with page citations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7">
          <Card header={<div className="flex items-center gap-2"><UploadCloud className="w-4 h-4 text-brand" /><h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wide">Document Ingestion</h2></div>}>
            <div className="border-2 border-dashed border-border-subtle rounded-lg p-8 text-center bg-surface-low/30 flex flex-col items-center">
              <div className="w-12 h-12 rounded-full bg-brand-light flex items-center justify-center text-brand mb-3">
                <UploadCloud className="w-6 h-6" />
              </div>
              <h3 className="font-heading font-semibold text-base text-text-primary mb-1">Drop your PDF here</h3>
              <p className="text-xs text-text-secondary max-w-sm mb-4">Upload research papers or reports up to {MAX_UPLOAD_SIZE_LABEL}.</p>
              <Button onClick={() => setIsUploadModalOpen(true)} variant="primary" size="sm" leftIcon={<FileText className="w-4 h-4" />}>Browse Files</Button>

            </div>
          </Card>
        </div>

        <div className="lg:col-span-5">
          <Card header={<div className="flex items-center gap-2"><Layers className="w-4 h-4 text-text-secondary" /><h2 className="font-heading text-sm font-semibold text-text-primary uppercase tracking-wide">How It Works</h2></div>}>
            <ol className="space-y-4 text-sm">
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-surface-container flex items-center justify-center text-xs font-bold shrink-0">1</div>
                <div><h4 className="font-semibold text-text-primary">Upload PDF</h4><p className="text-xs text-text-secondary">Validation &amp; page rendering.</p></div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-surface-container flex items-center justify-center text-xs font-bold shrink-0">2</div>
                <div><h4 className="font-semibold text-text-primary">Multimodal Indexing</h4><p className="text-xs text-text-secondary">Dense vectors, BM25 &amp; visual embeddings.</p></div>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-surface-container flex items-center justify-center text-xs font-bold shrink-0">3</div>
                <div><h4 className="font-semibold text-text-primary">Grounded Q&amp;A</h4><p className="text-xs text-text-secondary">Verifiable answers with citations.</p></div>
              </li>
            </ol>
          </Card>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-heading text-lg font-semibold text-text-primary">Recent Documents</h2>
          <Link href="/documents" className="inline-flex items-center gap-1 text-xs font-semibold text-brand hover:text-brand-hover">
            <span>View Library</span><ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {error && (
          <ErrorBanner
            title="Unable to load recent documents"
            message={error}
            onRetry={fetchRecentDocuments}
          />
        )}

        {!error && isLoading && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SkeletonCard />
            <SkeletonCard />
          </div>
        )}

        {!error && !isLoading && recentDocs.length === 0 && (
          <EmptyState
            icon={<Layers className="w-7 h-7 text-brand" />}
            title="No documents yet"
            description="Upload your first research paper or PDF document to begin extracting multimodal evidence and asking grounded questions."
            action={
              <Button
                variant="primary"
                size="md"
                onClick={() => setIsUploadModalOpen(true)}
                leftIcon={<UploadCloud className="w-4 h-4" />}
              >
                Upload document
              </Button>
            }
          />
        )}

        {!error && !isLoading && recentDocs.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recentDocs.map((doc) => (
              <Link key={doc.id} href={`/documents/${doc.id}`} className="bg-surface-elevated border border-border-subtle hover:border-brand/30 hover:shadow-xs transition-all rounded p-4 flex items-center justify-between group">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded bg-surface-low group-hover:bg-brand-light flex items-center justify-center text-text-secondary group-hover:text-brand shrink-0 transition-colors">
                    <FileText className="w-5 h-5" />
                  </div>
                  <div className="min-w-0">
                    <h4 className="text-sm font-medium text-text-primary group-hover:text-brand transition-colors truncate">{doc.name}</h4>
                    <p className="text-xs text-text-tertiary">{doc.pageCount} pages &bull; {doc.fileSizeFormatted} &bull; {doc.uploadedAt}</p>
                  </div>
                </div>
                <StatusBadge status={doc.status} size="sm" />
              </Link>
            ))}
          </div>
        )}
      </div>

      <UploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUploadSuccess={fetchRecentDocuments}
      />
    </div>
  );
}
