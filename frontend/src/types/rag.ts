/**
 * FieldSure AI — RAG Frontend Types
 *
 * Defines TypeScript contracts for backend API interactions,
 * query requests, grounded responses, citations, and document uploads.
 */

export interface SourceCitation {
  source: string;
  chunk_id: string | null;
  score: number | null;
  id?: string;
  label?: string;
  document?: string;
  section?: string | null;
  text?: string;
}

export interface StreamCitationsEvent {
  type: "citations";
  sources: SourceCitation[];
  status?: string;
  metadata?: QueryMetadata;
}

export interface StreamTokenEvent {
  type: "token";
  text: string;
}

export interface StreamDoneEvent {
  type: "done";
}

export interface StreamErrorEvent {
  type: "error";
  message: string;
}

export type StreamEvent =
  | StreamCitationsEvent
  | StreamTokenEvent
  | StreamDoneEvent
  | StreamErrorEvent;

export interface QueryMetadata {
  top_score?: number | null;
  chunks_used?: number | null;
  is_refused?: boolean | null;
  total_sources?: number | null;
  latency_ms?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceCitation[];
  status: "answered" | "refused_weak_context" | string;
  metadata?: QueryMetadata | null;
}

export interface QueryRequest {
  question: string;
}

export interface UploadSummary {
  document: string;
  chunks: number;
  indexed: number;
}

export interface UploadResponse {
  status: string;
  filename: string;
  summary: UploadSummary;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  latency_ms?: number;
}

export interface ChatInteraction {
  id: string;
  question: string;
  timestamp: string;
  response?: QueryResponse;
  error?: string;
  loading?: boolean;
}
