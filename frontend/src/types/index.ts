/**
 * DocuLens AI - Frontend TypeScript Definitions
 * Aligned with FastAPI backend schemas (/app/schemas/*)
 */

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ComponentStatus {
  status: "healthy" | "degraded" | "unhealthy";
  details?: string | null;
}

export interface ReadinessResponse {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  components: Record<string, ComponentStatus>;
}

export interface SystemHealthMetrics {
  status: "healthy" | "degraded" | "unhealthy";
  uptime_seconds: number;
  components: Record<string, ComponentStatus>;
}

export interface RequestTelemetryMetrics {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  status_codes: Record<string, number>;
  requests_by_endpoint: Record<string, number>;
}

export interface IngestionMetrics {
  ingestion_count: number;
  total_pages_processed: number;
  total_chunks_created: number;
  rendering_failures_count: number;
  avg_duration_ms?: number | null;
}

export interface RetrievalMetrics {
  total_queries: number;
  empty_queries: number;
  avg_latency_ms?: number | null;
  avg_result_count?: number | null;
}

export interface RerankingMetrics {
  total_queries: number;
  avg_latency_ms?: number | null;
}

export interface GenerationMetrics {
  total_queries: number;
  avg_latency_ms?: number | null;
  provider: string;
  model: string;
  timeout_count: number;
  rate_limit_count: number;
}

export interface PipelineTelemetryMetrics {
  ingestion: IngestionMetrics;
  retrieval: RetrievalMetrics;
  reranking: RerankingMetrics;
  generation: GenerationMetrics;
}

export interface AIQualityMetrics {
  total_answers_generated: number;
  grounded_answers_count: number;
  refusal_answers_count: number;
  valid_citations_count: number;
  rejected_citations_count: number;
}

export interface SystemMetricsResponse {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  timestamp: string;
  system_health: SystemHealthMetrics;
  request_telemetry: RequestTelemetryMetrics;
  pipeline_telemetry: PipelineTelemetryMetrics;
  ai_quality_metrics: AIQualityMetrics;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: string;
}

export interface ExtractedPage {
  page_number: number;
  text: string;
  char_count: number;
}

export interface DocumentMetadata {
  title?: string | null;
  author?: string | null;
  creation_date?: string | null;
  total_pages: number;
  file_size_bytes: number;
}

export interface ExtractionResult {
  document_id: string;
  metadata: DocumentMetadata;
  pages: ExtractedPage[];
}

export interface IndexingResult {
  document_id: string;
  total_pages: number;
  total_chunks: number;
  total_embeddings: number;
  status: string;
}

export interface DocumentListItemResponse {
  document_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: string;
  total_pages: number;
  uploaded_at: string | null;
}

export type DocumentStatus = "ready" | "processing" | "needs_attention" | "failed";

export interface DocumentItem {
  id: string;
  name: string;
  documentType?: string;
  pageCount: number;
  status: DocumentStatus;
  uploadedAt: string;
  uploadedAtISO?: string | null;
  fileSizeFormatted: string;
  sizeBytes?: number;
  description?: string;
  statusDetails?: string;
}

export type UploadProcessingState =
  | "idle"
  | "validating"
  | "uploading"
  | "processing"
  | "ready"
  | "failed";

export interface PipelineTelemetry {
  documentId?: string;
  filename?: string;
  sizeBytes?: number;
  totalPages?: number;
  totalChunks?: number;
  totalEmbeddings?: number;
  uploadDurationMs?: number;
  indexingDurationMs?: number;
  totalDurationMs?: number;
  indexingStatus?: string;
}

export type AnswerStyle = "concise" | "balanced" | "detailed";

export interface UserSettings {
  answerStyle: AnswerStyle;
  alwaysShowSources: boolean;
  highlightEvidence: boolean;
  showPipelineOnDemand: boolean;
}

export interface ApiError {
  message: string;
  status?: number;
  details?: unknown;
  code?: string;
}


export interface Citation {
  reference: string;
  rank: number;
  score: number;
  chunk_id?: string | null;
  document_id: string;
  page_number: number;
  chunk_index?: number | null;
  evidence_text?: string;
  text?: string | null;
  image_url?: string | null;
  retrieval_type?: string;
  sources?: string[];
  metadata?: Record<string, unknown>;
}

export interface QuestionRequest {
  question: string;
  top_k?: number;
  score_threshold?: number | null;
  answer_style?: AnswerStyle;
}

export interface PipelineStageTrace {
  stage_id: string;
  stage_name: string;
  order: number;
  status: "success" | "fallback" | "failed" | "skipped";
  duration_ms: number;
  input_count?: number | null;
  output_count?: number | null;
  error_category?: string | null;
  description: string;
  fallback_used: boolean;
  fallback_reason?: string | null;
  error_message?: string | null;
  details: Record<string, any>;
}

export interface PipelineSummary {
  total_duration_ms: number;
  retrieval_duration_ms?: number | null;
  reranking_duration_ms?: number | null;
  generation_duration_ms?: number | null;
  citation_validation_duration_ms?: number | null;
  total_stages: number;
  successful_stages: number;
  failed_stages: number;
  fallback_stages: number;
}

export interface PipelineTrace {
  pipeline_id: string;
  stages: PipelineStageTrace[];
  summary: PipelineSummary;
}

export interface GenerationResult {
  question: string;
  answer: string;
  document_id?: string | null;
  model: string;
  provider: string;
  citations: Citation[];
  is_grounded: boolean;
  usage?: Record<string, unknown>;
  pipeline_trace?: PipelineTrace | null;
}

export interface QAMessage {
  id: string;
  question: string;
  result?: GenerationResult;
  isLoading?: boolean;
  error?: string | null;
  timestamp: string;
}
