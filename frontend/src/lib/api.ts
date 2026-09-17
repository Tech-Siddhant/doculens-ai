/**
 * DocuLens AI - Centralized Frontend API Client
 *
 * Communicates exclusively with the FastAPI backend REST API.
 * Never connects directly to vector stores, databases, or third-party AI APIs.
 * Never stores or transmits secrets/API keys on the client side.
 */

import type {
  DocumentUploadResponse,
  DocumentListItemResponse,
  ExtractionResult,
  IndexingResult,
  DocumentItem,
  DocumentStatus,
  HealthResponse,
  ApiError,
  QuestionRequest,
  GenerationResult,
  Citation,
} from "../types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const API_PREFIX = "/api/v1";

/** Authoritative backend upload limit (10 MB = 10,485,760 bytes) */
export const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024;
export const MAX_UPLOAD_SIZE_LABEL = "10 MB";

export function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${val.toFixed(val >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function formatMilliseconds(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)} ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatDate(isoDate: string | null): string {
  if (!isoDate) return "—";
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) return "—";

    const diffSec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (diffSec >= 0 && diffSec < 60) return "Just now";
    if (diffSec >= 60 && diffSec < 3600) return `${Math.floor(diffSec / 60)} mins ago`;
    if (diffSec >= 3600 && diffSec < 86400) return `${Math.floor(diffSec / 3600)} hours ago`;
    if (diffSec >= 86400 && diffSec < 172800) return "Yesterday";

    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: d.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  } catch {
    return "—";
  }
}

export function mapBackendDocToItem(doc: DocumentListItemResponse): DocumentItem {
  let status: DocumentStatus = "ready";
  const rawStatus = (doc.status || "").toLowerCase();
  if (rawStatus === "failed" || rawStatus === "error" || rawStatus === "corrupted") {
    status = "failed";
  } else if (rawStatus === "processing" || rawStatus === "indexing" || rawStatus === "extracting") {
    status = "processing";
  } else if (rawStatus === "needs_attention" || rawStatus === "degraded" || rawStatus === "warning") {
    status = "needs_attention";
  } else {
    status = "ready";
  }

  let name = doc.filename || `${doc.document_id}.pdf`;
  if (!name.toLowerCase().endsWith(".pdf")) {
    name = `${name}.pdf`;
  }

  return {
    id: doc.document_id,
    name,
    documentType: "PDF",
    pageCount: doc.total_pages || 0,
    status,
    uploadedAt: formatDate(doc.uploaded_at),
    uploadedAtISO: doc.uploaded_at,
    fileSizeFormatted: formatBytes(doc.size_bytes),
    sizeBytes: doc.size_bytes,
  };
}

export function getRecommendedActionForError(status?: number, message?: string): {
  code: string;
  recommendation: string;
} {
  const msg = (message || "").toLowerCase();

  if (status === 413 || msg.includes("exceeds") || msg.includes("size")) {
    return {
      code: "ERR_FILE_OVERSIZED",
      recommendation: "Compress the PDF or select a document smaller than 10 MB.",
    };
  }
  if (status === 400 && (msg.includes("magic") || msg.includes("corrupt") || msg.includes("header") || msg.includes("unreadable"))) {
    return {
      code: "ERR_PDF_CORRUPTED_OR_INVALID",
      recommendation: "Re-export PDF via standard 'Print to PDF' or export tool to ensure valid PDF structure.",
    };
  }
  if (status === 504 || msg.includes("timed out") || msg.includes("timeout")) {
    return {
      code: "ERR_GATEWAY_TIMEOUT",
      recommendation: "Generation request timed out. Please try again with a more specific question or check network connectivity.",
    };
  }
  if (status === 429 || msg.includes("rate limit") || msg.includes("too many requests")) {
    return {
      code: "ERR_RATE_LIMIT",
      recommendation: "Rate limit exceeded on the AI provider. Please wait a few moments before sending another request.",
    };
  }
  if (status === 503 || msg.includes("temporarily unavailable") || msg.includes("service unavailable")) {
    return {
      code: "ERR_SERVICE_UNAVAILABLE",
      recommendation: "The AI provider service is temporarily unavailable. Please retry shortly.",
    };
  }
  if (status === 502 || msg.includes("bad gateway") || msg.includes("authentication failed")) {
    return {
      code: "ERR_BAD_GATEWAY",
      recommendation: "Upstream AI provider error or invalid credentials. Please check backend API key configuration.",
    };
  }
  if (msg.includes("password") || msg.includes("encrypt")) {
    return {
      code: "ERR_PDF_ENCRYPTED",
      recommendation: "Remove password protection or encryption before uploading.",
    };
  }
  if (status === 0 || msg.includes("connect") || msg.includes("server")) {
    return {
      code: "ERR_NETWORK_DISCONNECTED",
      recommendation: "Verify that the DocuLens AI backend service is running and accessible.",
    };
  }
  return {
    code: `ERR_PROCESSING_HTTP_${status || "GENERIC"}`,
    recommendation: "Check file integrity and try uploading again. If the issue persists, review backend logs.",
  };
}

export class DocuLensApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
  }

  /**
   * Generic centralized request handler with consistent error handling.
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${API_PREFIX}${endpoint}`;

    const headers = new Headers(options.headers || {});
    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json");
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        let errorMessage = `Request failed with status ${response.status}`;
        try {
          const errorData = await response.json();
          if (typeof errorData?.detail === "string") {
            errorMessage = errorData.detail;
          } else if (Array.isArray(errorData?.detail)) {
            errorMessage = errorData.detail
              .map((d: { msg?: string }) => d.msg || "Invalid request")
              .join(", ");
          } else if (errorData?.message) {
            errorMessage = errorData.message;
          }
        } catch {
          if (response.statusText) {
            errorMessage = response.statusText;
          }
        }

        const errMeta = getRecommendedActionForError(response.status, errorMessage);
        const apiError: ApiError = {
          message: errorMessage,
          status: response.status,
          code: errMeta.code,
        };
        throw apiError;
      }

      return (await response.json()) as T;
    } catch (error: unknown) {
      if ((error as ApiError).message) {
        throw error;
      }
      const apiError: ApiError = {
        message: "Unable to connect to the DocuLens AI server. Please verify the backend is running.",
        status: 0,
        code: "ERR_NETWORK_DISCONNECTED",
      };
      throw apiError;
    }
  }

  /**
   * Fetch backend system health status.
   * Endpoint: GET /api/v1/health
   */
  async checkHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  /**
   * Fetch all stored documents.
   * Endpoint: GET /api/v1/documents
   */
  async getDocuments(): Promise<DocumentListItemResponse[]> {
    return this.request<DocumentListItemResponse[]>("/documents");
  }

  /**
   * Fetch a single document by ID.
   * Endpoint: GET /api/v1/documents/{document_id}
   */
  async getDocument(documentId: string): Promise<DocumentListItemResponse> {
    return this.request<DocumentListItemResponse>(`/documents/${documentId}`);
  }

  /**
   * Upload a PDF document to the ingestion storage.
   * Endpoint: POST /api/v1/documents/upload
   */
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return this.request<DocumentUploadResponse>("/documents/upload", {
      method: "POST",
      body: formData,
    });
  }

  /**
   * Trigger text and layout extraction for an uploaded document.
   * Endpoint: POST /api/v1/documents/{document_id}/extract
   */
  async extractDocument(documentId: string): Promise<ExtractionResult> {
    return this.request<ExtractionResult>(`/documents/${encodeURIComponent(documentId)}/extract`, {
      method: "POST",
    });
  }

  /**
   * Trigger indexing (chunking, embedding, vector & BM25 indexing) for a document.
   * Endpoint: POST /api/v1/documents/{document_id}/index
   */
  async indexDocument(
    documentId: string,
    chunkSize?: number,
    chunkOverlap?: number
  ): Promise<IndexingResult> {
    const params = new URLSearchParams();
    if (chunkSize !== undefined) params.set("chunk_size", String(chunkSize));
    if (chunkOverlap !== undefined) params.set("chunk_overlap", String(chunkOverlap));
    const qs = params.toString() ? `?${params.toString()}` : "";

    return this.request<IndexingResult>(
      `/documents/${encodeURIComponent(documentId)}/index${qs}`,
      {
        method: "POST",
      }
    );
  }
  /**
   * Submit a question for a specific document and receive a grounded answer with citations.
   * Endpoint: POST /api/v1/documents/{document_id}/ask
   */
  async askDocument(
    documentId: string,
    req: QuestionRequest
  ): Promise<GenerationResult> {
    return this.request<GenerationResult>(
      `/documents/${encodeURIComponent(documentId)}/ask`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req),
      }
    );
  }

  /**
   * Submit a question across documents and receive a grounded answer.
   * Endpoint: POST /api/v1/documents/ask
   */
  async askCollection(
    req: QuestionRequest,
    documentId?: string
  ): Promise<GenerationResult> {
    const qs = documentId ? `?document_id=${encodeURIComponent(documentId)}` : "";
    return this.request<GenerationResult>(`/documents/ask${qs}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(req),
    });
  }

  /**
   * Get the direct URL for a rendered document page image.
   * Endpoint: GET /api/v1/documents/{document_id}/pages/{page_number}/image
   */
  getPageImageUrl(documentId: string, pageNumber: number): string {
    return `${this.baseUrl}${API_PREFIX}/documents/${encodeURIComponent(documentId)}/pages/${pageNumber}/image`;
  }
}

// Global singleton instance for use throughout client components
export const apiClient = new DocuLensApiClient();

