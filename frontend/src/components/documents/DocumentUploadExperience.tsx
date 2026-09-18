"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  X,
  RefreshCw,
  Terminal,
  ShieldCheck,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Database,
  Cpu,
  FileSearch,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import {
  apiClient,
  formatBytes,
  formatMilliseconds,
  MAX_UPLOAD_SIZE_BYTES,
  MAX_UPLOAD_SIZE_LABEL,
  getRecommendedActionForError,
} from "@/lib/api";
import {
  DocumentUploadResponse,
  IndexingResult,
  PipelineTelemetry,
  UploadProcessingState,
  ApiError,
} from "@/types";

interface DocumentUploadExperienceProps {
  isModal?: boolean;
  onCloseModal?: () => void;
  onUploadSuccess?: (documentId: string) => void;
}

export const DocumentUploadExperience: React.FC<DocumentUploadExperienceProps> = ({
  isModal = false,
  onCloseModal,
  onUploadSuccess,
}) => {
  const [state, setState] = useState<UploadProcessingState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [showTelemetry, setShowTelemetry] = useState(false);
  const [showErrorDetails, setShowErrorDetails] = useState(false);

  // In-flight progress timing
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Telemetry & Results data
  const [telemetry, setTelemetry] = useState<PipelineTelemetry | null>(null);
  const [uploadResult, setUploadResult] = useState<DocumentUploadResponse | null>(null);
  const [indexingResult, setIndexingResult] = useState<IndexingResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);

  // System status check
  const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking");
  const [backendVersion, setBackendVersion] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let mounted = true;
    apiClient
      .checkHealth()
      .then((res) => {
        if (mounted) {
          setBackendStatus("online");
          setBackendVersion(res.version);
        }
      })
      .catch(() => {
        if (mounted) {
          setBackendStatus("offline");
        }
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Timer effect during active processing
  useEffect(() => {
    if (state === "uploading" || state === "processing") {
      setElapsedSeconds(0);
      const start = Date.now();
      timerRef.current = setInterval(() => {
        setElapsedSeconds(Math.floor((Date.now() - start) / 1000));
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [state]);

  const resetFlow = useCallback(() => {
    setState("idle");
    setFile(null);
    setTelemetry(null);
    setUploadResult(null);
    setIndexingResult(null);
    setErrorMessage(null);
    setErrorCode(null);
    setErrorStatus(null);
    setShowErrorDetails(false);
    setElapsedSeconds(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  const validateAndProcessFile = useCallback(
    async (selectedFile: File) => {
      // 1. Client-side format check
      const fileNameLower = selectedFile.name.toLowerCase();
      if (!fileNameLower.endsWith(".pdf") && selectedFile.type !== "application/pdf") {
        setState("failed");
        setFile(selectedFile);
        setErrorMessage("Invalid file format. Only PDF documents (.pdf) are supported.");
        setErrorCode("ERR_INVALID_FILE_TYPE");
        setErrorStatus(400);
        return;
      }

      // 2. Client-side empty file check
      if (selectedFile.size === 0) {
        setState("failed");
        setFile(selectedFile);
        setErrorMessage("The selected file is empty (0 bytes). Please choose a valid PDF document.");
        setErrorCode("ERR_EMPTY_FILE");
        setErrorStatus(400);
        return;
      }

      // 3. Client-side authoritative size check (10 MB backend limit)
      if (selectedFile.size > MAX_UPLOAD_SIZE_BYTES) {
        setState("failed");
        setFile(selectedFile);
        setErrorMessage(
          `File size (${formatBytes(selectedFile.size)}) exceeds the maximum allowed limit of ${MAX_UPLOAD_SIZE_LABEL}.`
        );
        setErrorCode("ERR_FILE_OVERSIZED");
        setErrorStatus(413);
        return;
      }

      // Validated file: proceed with Upload + Ingestion state machine
      setFile(selectedFile);
      setErrorMessage(null);
      setErrorCode(null);
      setErrorStatus(null);
      setShowErrorDetails(false);
      setState("uploading");

      const uploadStart = performance.now();

      try {
        // Step 1: Upload to backend
        const uploadRes = await apiClient.uploadDocument(selectedFile);
        const uploadEnd = performance.now();
        const uploadDurationMs = Math.round(uploadEnd - uploadStart);

        setUploadResult(uploadRes);
        setState("processing");

        // Step 2: Index document (chunking, embedding, vector store upsert, BM25 indexing)
        const indexingStart = performance.now();
        const indexRes = await apiClient.indexDocument(uploadRes.document_id);
        const indexingEnd = performance.now();
        const indexingDurationMs = Math.round(indexingEnd - indexingStart);
        const totalDurationMs = uploadDurationMs + indexingDurationMs;

        setIndexingResult(indexRes);
        setTelemetry({
          documentId: uploadRes.document_id,
          filename: uploadRes.filename,
          sizeBytes: uploadRes.size_bytes,
          totalPages: indexRes.total_pages,
          totalChunks: indexRes.total_chunks,
          totalEmbeddings: indexRes.total_embeddings,
          uploadDurationMs,
          indexingDurationMs,
          totalDurationMs,
          indexingStatus: indexRes.status,
        });

        setState("ready");
        if (onUploadSuccess) {
          onUploadSuccess(uploadRes.document_id);
        }
      } catch (err: unknown) {
        const apiErr = err as ApiError;
        const msg =
          apiErr.message ||
          "An error occurred while uploading or indexing the document. Please try again.";
        const status = apiErr.status || 0;
        const errorMeta = getRecommendedActionForError(status, msg);

        setState("failed");
        setErrorMessage(msg);
        setErrorCode(apiErr.code || errorMeta.code);
        setErrorStatus(status);
      }
    },
    [onUploadSuccess]
  );

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (state === "uploading" || state === "processing") return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  return (
    <div className="w-full space-y-6">
      {/* 1. Header with Breadcrumbs & Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          {!isModal && (
            <div className="flex items-center gap-1.5 text-xs text-text-tertiary font-mono mb-1">
              <Link href="/documents" className="hover:text-text-primary transition-colors">
                Documents
              </Link>
              <span>/</span>
              <span className="text-text-secondary">Add a document</span>
            </div>
          )}
          <h1 className="font-heading text-xl sm:text-2xl font-bold tracking-tight text-text-primary">
            Add a document
          </h1>
          <p className="text-text-secondary text-xs sm:text-sm mt-0.5">
            Upload a PDF to read, search, and ask questions about it.
          </p>
        </div>

        {/* Status Pill & Telemetry Toggle */}
        <div className="flex items-center gap-3">
          <div
            className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border border-border-subtle bg-surface-elevated font-mono"
            title={`API Status: ${backendStatus}`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus === "online"
                  ? "bg-status-ready"
                  : backendStatus === "offline"
                  ? "bg-status-failed"
                  : "bg-status-processing animate-pulse"
              }`}
            />
            <span className="text-text-secondary text-[11px]">
              {backendStatus === "online"
                ? `System Ready (API v${backendVersion || "0.1.0"})`
                : backendStatus === "offline"
                ? "Backend Offline"
                : "Checking System..."}
            </span>
          </div>

          <button
            type="button"
            onClick={() => setShowTelemetry((prev) => !prev)}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border transition-colors font-mono uppercase tracking-wider ${
              showTelemetry
                ? "bg-brand text-white border-brand shadow-xs"
                : "bg-surface-elevated text-text-secondary border-border-subtle hover:text-text-primary hover:border-border-strong"
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>{showTelemetry ? "Hide Technical Process" : "Show Technical Process"}</span>
          </button>

          {isModal && onCloseModal && (
            <button
              type="button"
              onClick={onCloseModal}
              disabled={state === "uploading" || state === "processing"}
              className="p-1 text-text-tertiary hover:text-text-primary rounded disabled:opacity-40"
              aria-label="Close modal"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* 2. Upload Dropzone (Visible in IDLE state) */}
      {state === "idle" && (
        <section
          aria-label="Upload PDF dropzone"
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          className={`relative border-2 border-dashed rounded-lg p-8 sm:p-12 text-center cursor-pointer transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-2 ${
            isDragging
              ? "border-brand bg-brand-light/40 shadow-inner"
              : "border-border-subtle bg-surface-elevated hover:bg-surface-low hover:border-border-strong"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            onChange={handleFileInputChange}
            className="hidden"
            aria-label="Select PDF document"
          />

          <div className="flex flex-col items-center max-w-md mx-auto pointer-events-none">
            <div className="w-14 h-14 rounded-full bg-brand-light flex items-center justify-center text-brand mb-4 shadow-xs">
              <UploadCloud className="w-7 h-7" />
            </div>

            <h2 className="font-heading font-semibold text-base sm:text-lg text-text-primary mb-1">
              Drop your PDF here
            </h2>
            <p className="text-xs sm:text-sm text-text-secondary mb-4">
              or click to browse your local files
            </p>

            <Button
              type="button"
              variant="outline"
              size="sm"
              className="pointer-events-auto shadow-xs"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Choose a file
            </Button>

            <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-xs text-text-tertiary mt-5 font-mono">
              <span className="text-text-secondary font-medium">
                PDF documents up to {MAX_UPLOAD_SIZE_LABEL}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-status-ready" />
                Encrypted &amp; isolated index
              </span>
              <span>•</span>
              <a
                href="#limits"
                onClick={(e) => e.stopPropagation()}
                className="text-brand hover:underline underline-offset-2 pointer-events-auto"
              >
                Supported limits
              </a>
            </div>
          </div>
        </section>
      )}

      {/* 3. In-Flight Processing Card (UPLOADING or PROCESSING state) */}
      {(state === "uploading" || state === "processing") && (
        <section
          aria-label="Active processing state"
          className="bg-surface-elevated border border-border-subtle rounded-lg overflow-hidden shadow-xs"
        >
          {/* File Card Header */}
          <div className="p-4 sm:p-5 border-b border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-surface-low/30">
            <div className="flex items-start gap-3 min-w-0">
              <div className="w-10 h-10 rounded bg-brand-light border border-blue-100 flex items-center justify-center text-brand shrink-0">
                <FileText className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="font-heading font-semibold text-sm sm:text-base text-text-primary truncate">
                    {file?.name || "Document.pdf"}
                  </h3>
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-mono font-medium bg-amber-50 text-amber-800 border border-amber-200">
                    <span className="w-1.5 h-1.5 rounded-full bg-status-processing animate-pulse" />
                    {state === "uploading"
                      ? "Uploading to storage..."
                      : "Indexing document..."}
                  </span>
                </div>
                <p className="font-mono text-xs text-text-tertiary mt-0.5">
                  {file ? formatBytes(file.size) : "—"} · IN-FLIGHT ({elapsedSeconds}s elapsed)
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 self-end sm:self-auto shrink-0">
              <button
                type="button"
                onClick={() => setShowTelemetry((prev) => !prev)}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-mono uppercase bg-surface-elevated border border-border-subtle text-text-secondary hover:text-text-primary rounded hover:border-border-strong transition-colors"
                title="Toggle technical process telemetry"
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>{showTelemetry ? "Hide Technical Process" : "Show Technical Process"}</span>
              </button>
              <button
                type="button"
                onClick={resetFlow}
                className="p-1.5 text-text-tertiary hover:text-status-failed hover:bg-rose-50 rounded transition-colors"
                title="Cancel processing"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Indeterminate Progress Indicator */}
          <div className="px-4 sm:px-5 pt-4 pb-2">
            <div className="flex items-center justify-between font-mono text-xs text-text-secondary mb-2">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-status-processing animate-ping" />
                {state === "uploading"
                  ? "STAGE 1 OF 4: UPLOADING & VERIFYING PAYLOAD"
                  : "STAGE 2-3 OF 4: CHUNKING & DENSE VECTORIZATION"}
              </span>
              <span className="text-text-tertiary">PROCESSING</span>
            </div>
            {/* Smooth animated indeterminate progress bar (No fake percentages) */}
            <div className="w-full h-1.5 bg-surface-container rounded-full overflow-hidden relative">
              <div className="h-full bg-brand rounded-full animate-indeterminate" />
            </div>
          </div>

          {/* 4-Step Progressive Timeline */}
          <div className="p-4 sm:p-5 pt-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Step 1 */}
              <div className="p-3 bg-surface-elevated border border-border-subtle rounded flex items-start gap-2.5">
                <div
                  className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center mt-0.5 text-xs ${
                    state === "processing"
                      ? "bg-emerald-50 text-status-ready border border-emerald-200"
                      : "bg-amber-50 text-status-processing border border-amber-200 animate-spin"
                  }`}
                >
                  {state === "processing" ? (
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  ) : (
                    <RefreshCw className="w-3 h-3" />
                  )}
                </div>
                <div className="min-w-0">
                  <p className="font-heading font-medium text-xs text-text-primary">
                    1. Reading document
                  </p>
                  <p className="text-[11px] text-text-tertiary mt-0.5 font-mono">
                    {state === "processing" ? "PDF format verified" : "Uploading payload..."}
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div
                className={`p-3 border rounded flex items-start gap-2.5 ${
                  state === "processing"
                    ? "bg-surface-elevated border-border-subtle"
                    : "bg-surface-low/50 border-border-subtle opacity-60"
                }`}
              >
                <div
                  className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center mt-0.5 text-xs ${
                    state === "processing"
                      ? "bg-amber-50 text-status-processing border border-amber-200 animate-spin"
                      : "border border-border-subtle text-text-tertiary"
                  }`}
                >
                  {state === "processing" ? (
                    <RefreshCw className="w-3 h-3" />
                  ) : (
                    <span className="font-mono text-[10px]">2</span>
                  )}
                </div>
                <div className="min-w-0">
                  <p className="font-heading font-medium text-xs text-text-primary">
                    2. Extracting content
                  </p>
                  <p className="text-[11px] text-text-tertiary mt-0.5 font-mono">
                    {state === "processing" ? "Parsing layout & text..." : "Queued"}
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div
                className={`p-3 border rounded flex items-start gap-2.5 ${
                  state === "processing"
                    ? "bg-surface-elevated border-border-subtle"
                    : "bg-surface-low/50 border-border-subtle opacity-60"
                }`}
              >
                <div
                  className={`w-5 h-5 shrink-0 rounded-full flex items-center justify-center mt-0.5 text-xs ${
                    state === "processing"
                      ? "bg-blue-50 text-brand border border-blue-200"
                      : "border border-border-subtle text-text-tertiary"
                  }`}
                >
                  {state === "processing" ? (
                    <Database className="w-3 h-3" />
                  ) : (
                    <span className="font-mono text-[10px]">3</span>
                  )}
                </div>
                <div className="min-w-0">
                  <p className="font-heading font-medium text-xs text-text-primary">
                    3. Preparing search
                  </p>
                  <p className="text-[11px] text-text-tertiary mt-0.5 font-mono">
                    {state === "processing" ? "Generating embeddings..." : "Queued"}
                  </p>
                </div>
              </div>

              {/* Step 4 */}
              <div className="p-3 bg-surface-low/50 border border-border-subtle rounded flex items-start gap-2.5 opacity-60">
                <div className="w-5 h-5 shrink-0 rounded-full border border-border-subtle text-text-tertiary flex items-center justify-center mt-0.5 text-[10px] font-mono">
                  4
                </div>
                <div className="min-w-0">
                  <p className="font-heading font-medium text-xs text-text-tertiary">
                    4. Finalizing
                  </p>
                  <p className="text-[11px] text-text-tertiary mt-0.5 font-mono">Queued</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* 4. Ready State Card (READY state) */}
      {state === "ready" && (
        <section
          aria-label="Document ready"
          className="bg-surface-elevated border border-border-subtle border-l-4 border-l-status-ready rounded-lg p-5 sm:p-6 shadow-xs"
        >
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200 uppercase tracking-wider">
                  <CheckCircle2 className="w-3.5 h-3.5 text-status-ready" />
                  Document ready
                </span>
                {telemetry?.totalDurationMs && (
                  <span className="font-mono text-xs text-text-tertiary uppercase">
                    Finished in {formatMilliseconds(telemetry.totalDurationMs)}
                  </span>
                )}
              </div>

              <h2 className="font-heading text-lg sm:text-xl font-bold text-text-primary tracking-tight">
                {file?.name || uploadResult?.filename || "Document.pdf"}
              </h2>

              <p className="text-xs sm:text-sm text-text-secondary max-w-2xl">
                Your document is processed and verified. You can now ask questions about it or
                inspect grounded citations directly in the workspace.
              </p>

              <div className="flex flex-wrap items-center gap-3 pt-2 font-mono text-xs text-text-secondary">
                <span className="flex items-center gap-1 bg-surface-low px-2 py-0.5 rounded border border-border-subtle">
                  <FileText className="w-3.5 h-3.5 text-text-tertiary" />
                  {indexingResult?.total_pages || 0} pages
                </span>
                <span className="flex items-center gap-1 bg-surface-low px-2 py-0.5 rounded border border-border-subtle">
                  <Cpu className="w-3.5 h-3.5 text-text-tertiary" />
                  {indexingResult?.total_chunks || 0} chunks
                </span>
                <span className="flex items-center gap-1 bg-surface-low px-2 py-0.5 rounded border border-border-subtle">
                  <Database className="w-3.5 h-3.5 text-text-tertiary" />
                  {indexingResult?.total_embeddings || 0} vectors
                </span>
                <span className="text-text-tertiary">
                  {uploadResult?.size_bytes ? formatBytes(uploadResult.size_bytes) : "—"}
                </span>
              </div>
            </div>

            {/* Quick Action CTAs */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5 shrink-0 pt-2 lg:pt-0">
              <Button
                variant="outline"
                size="sm"
                onClick={resetFlow}
                leftIcon={<UploadCloud className="w-4 h-4" />}
              >
                Upload another
              </Button>
              <Link href="/documents">
                <Button variant="secondary" size="sm" leftIcon={<FileSearch className="w-4 h-4" />}>
                  View in Library
                </Button>
              </Link>
              <Link href={`/documents/${uploadResult?.document_id || ""}`}>
                <Button
                  variant="primary"
                  size="sm"
                  leftIcon={<Sparkles className="w-4 h-4" />}
                  rightIcon={<ArrowRight className="w-4 h-4" />}
                >
                  Open in Workspace
                </Button>
              </Link>
            </div>
          </div>
        </section>
      )}

      {/* 5. Error State Card (FAILED state) */}
      {state === "failed" && (
        <section
          aria-label="Upload error"
          className="bg-surface-elevated border border-border-subtle border-l-4 border-l-status-failed rounded-lg p-5 sm:p-6 shadow-xs space-y-4"
        >
          <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
            <div className="space-y-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-rose-50 text-rose-800 border border-rose-200 uppercase tracking-wider">
                  <AlertCircle className="w-3.5 h-3.5 text-status-failed" />
                  Processing Failed
                </span>
              </div>
              <h2 className="font-heading text-base sm:text-lg font-bold text-text-primary">
                {file?.name || "Document.pdf"} — We couldn&apos;t process this document
              </h2>
              <p className="text-xs sm:text-sm text-text-secondary">
                {errorMessage ||
                  "The file could not be parsed into readable text or visual pages. It may be password-protected, corrupted, or unsupported."}
              </p>
            </div>

            <div className="flex items-center gap-2 shrink-0 self-start sm:self-auto flex-wrap">
              <Button
                variant="outline"
                size="sm"
                onClick={resetFlow}
                leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
              >
                Retry upload
              </Button>
              <button
                type="button"
                onClick={() => setShowTelemetry((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-mono rounded border transition-colors ${
                  showTelemetry
                    ? "bg-brand text-white border-brand shadow-xs"
                    : "bg-surface-elevated text-text-secondary border-border-subtle hover:text-text-primary hover:bg-surface-low"
                }`}
                title="Toggle technical process telemetry"
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>{showTelemetry ? "Hide Technical Process" : "Show Technical Process"}</span>
              </button>
              <button
                type="button"
                onClick={() => setShowErrorDetails((prev) => !prev)}
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-mono text-text-secondary hover:text-text-primary rounded hover:bg-surface-low transition-colors"
              >
                <span>{showErrorDetails ? "Hide details" : "View details"}</span>
                {showErrorDetails ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          {/* Diagnostic Details Panel (Progressive Disclosure) */}
          {showErrorDetails && (
            <div className="bg-surface-low border border-border-subtle rounded p-3.5 font-mono text-xs space-y-2">
              <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                <span className="text-text-tertiary">ERROR_CODE:</span>
                <span className="text-text-primary font-semibold">
                  {errorCode || "ERR_GENERIC_UPLOAD_FAILURE"}
                </span>
              </div>
              {errorStatus !== null && (
                <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                  <span className="text-text-tertiary">HTTP_STATUS:</span>
                  <span className="text-text-primary">{errorStatus}</span>
                </div>
              )}
              <div className="flex flex-col sm:flex-row sm:justify-between gap-1">
                <span className="text-text-tertiary">RECOMMENDATION:</span>
                <span className="text-text-secondary text-right sm:max-w-md">
                  {getRecommendedActionForError(errorStatus || undefined, errorMessage || undefined)
                    .recommendation}
                </span>
              </div>
            </div>
          )}
        </section>
      )}

      {/* 6. Observable Backend Telemetry Panel (Progressive Disclosure) */}
      {showTelemetry && (
        <section
          aria-label="Observable backend ingestion telemetry"
          className="bg-surface-elevated border border-border-subtle rounded-lg p-5 space-y-4 shadow-xs"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-border-subtle gap-2">
            <div>
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-brand" />
                <h3 className="font-heading font-semibold text-xs text-text-primary uppercase tracking-wider font-mono">
                  Technical Ingestion Pipeline
                </h3>
              </div>
              <p className="text-[11px] text-text-tertiary font-mono mt-0.5">
                Observable backend ingestion telemetry &amp; vector index metadata
              </p>
            </div>
            <div className="font-mono text-[11px] text-text-tertiary flex items-center gap-2">
              <span>
                MODE:{" "}
                <code className="text-text-primary bg-surface-low px-1.5 py-0.5 rounded">
                  SYNCHRONOUS
                </code>
              </span>
              <span>·</span>
              <span className="text-status-ready font-semibold">ZERO RETENTION</span>
            </div>
          </div>

          {/* Telemetry Stage Rows */}
          <div className="space-y-1.5 font-mono text-xs">
            {/* Stage 1: Document Upload / ID */}
            <div className="flex items-center justify-between p-2 bg-surface-low/50 rounded border-l-2 border-status-ready">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-ready" />
                <span className="text-text-primary font-medium">DOCUMENT REGISTRATION</span>
              </div>
              <div className="flex items-center gap-3 text-text-secondary text-[11px]">
                <span className="truncate max-w-[140px] sm:max-w-none">
                  {uploadResult?.document_id || telemetry?.documentId || "Awaiting file..."}
                </span>
                {telemetry?.uploadDurationMs && (
                  <span className="text-text-primary font-semibold">
                    {formatMilliseconds(telemetry.uploadDurationMs)}
                  </span>
                )}
              </div>
            </div>

            {/* Stage 2: Payload Storage */}
            <div className="flex items-center justify-between p-2 bg-surface-low/50 rounded border-l-2 border-status-ready">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-ready" />
                <span className="text-text-primary font-medium">STORAGE &amp; MIME VALIDATION</span>
              </div>
              <div className="flex items-center gap-3 text-text-secondary text-[11px]">
                <span>
                  {uploadResult?.size_bytes
                    ? `${formatBytes(uploadResult.size_bytes)} (${uploadResult.size_bytes} bytes)`
                    : file
                    ? formatBytes(file.size)
                    : "—"}
                </span>
                <span className="text-status-ready font-medium">application/pdf</span>
              </div>
            </div>

            {/* Stage 3: Extraction & Page Verification */}
            <div className="flex items-center justify-between p-2 bg-surface-low/50 rounded border-l-2 border-status-ready">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-ready" />
                <span className="text-text-primary font-medium">LAYOUT &amp; TEXT EXTRACTION</span>
              </div>
              <div className="flex items-center gap-3 text-text-secondary text-[11px]">
                <span>
                  {indexingResult?.total_pages
                    ? `${indexingResult.total_pages} pages extracted`
                    : "PyPDF + pdfplumber"}
                </span>
                <span className="text-text-primary font-semibold">
                  {state === "ready" ? "VERIFIED" : state === "processing" ? "IN-FLIGHT" : "QUEUED"}
                </span>
              </div>
            </div>

            {/* Stage 4: Semantic Chunking & Vector Upsert */}
            <div className="flex items-center justify-between p-2 bg-surface-low/50 rounded border-l-2 border-brand">
              <div className="flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-brand" />
                <span className="text-text-primary font-medium">VECTOR STORE &amp; BM25 RETRIEVER</span>
              </div>
              <div className="flex items-center gap-3 text-text-secondary text-[11px]">
                <span>
                  {indexingResult?.total_chunks
                    ? `${indexingResult.total_chunks} chunks · ${indexingResult.total_embeddings} vectors`
                    : "Qdrant + BM25"}
                </span>
                {telemetry?.indexingDurationMs && (
                  <span className="text-brand font-semibold">
                    {formatMilliseconds(telemetry.indexingDurationMs)}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between font-mono text-[10px] text-text-tertiary border-t border-border-subtle gap-1">
            <span>PIPELINE: Fast Ingestion · Dense (text-embedding-004) &amp; BM25</span>
            <span>STORAGE: data/uploads · Tenant-isolated</span>
          </div>
        </section>
      )}

      {/* 7. Supported Limits & Specifications Banner */}
      <section
        id="limits"
        aria-label="Supported limits and specifications"
        className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2"
      >
        <div className="p-4 bg-surface-elevated border border-border-subtle rounded-lg">
          <div className="font-mono text-xs uppercase text-text-tertiary mb-1">
            Index Capacity
          </div>
          <p className="font-heading text-sm font-semibold text-text-primary">
            Up to {MAX_UPLOAD_SIZE_LABEL} per file
          </p>
          <p className="text-xs text-text-secondary mt-1">
            Multi-page research papers, technical documentation, and complex reports.
          </p>
        </div>

        <div className="p-4 bg-surface-elevated border border-border-subtle rounded-lg">
          <div className="font-mono text-xs uppercase text-text-tertiary mb-1">
            Multimodal Grounding
          </div>
          <p className="font-heading text-sm font-semibold text-text-primary">
            Hybrid Dense + Lexical Retrieval
          </p>
          <p className="text-xs text-text-secondary mt-1">
            Retrieves exact evidence chunks and cites page coordinates for verifiable answers.
          </p>
        </div>

        <div className="p-4 bg-surface-elevated border border-border-subtle rounded-lg">
          <div className="font-mono text-xs uppercase text-text-tertiary mb-1">
            Privacy &amp; Isolation
          </div>
          <p className="font-heading text-sm font-semibold text-text-primary">
            Tenant-Isolated Storage
          </p>
          <p className="text-xs text-text-secondary mt-1">
            Zero public training. Your document data is private and securely staged.
          </p>
        </div>
      </section>
    </div>
  );
};
