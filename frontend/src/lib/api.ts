/**
 * FieldSure AI — RAG API Client
 *
 * Provides typed functions to communicate with the FastAPI backend:
 * - askQuestion: POST /query
 * - checkHealth: GET /health
 * - uploadDocument: POST /documents
 */

import { HealthStatus, QueryResponse, UploadResponse } from "../types/rag";

export const DEFAULT_API_URL =
  process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8001";

/**
 * Retrieve the active API base URL (checks localStorage if customized).
 */
export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const saved = localStorage.getItem("fieldsure_api_url");
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, "");
    }
  }
  return DEFAULT_API_URL.replace(/\/+$/, "");
}

/**
 * Save custom API base URL to localStorage.
 */
export function setApiBaseUrl(url: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem("fieldsure_api_url", url.trim().replace(/\/+$/, ""));
  }
}

/**
 * Query the backend RAG pipeline.
 *
 * @param question Clinical or operational question to ask.
 * @param baseUrl Optional override for API base URL.
 * @returns Grounded answer response with sources and status.
 */
export async function askQuestion(
  question: string,
  baseUrl: string = getApiBaseUrl()
): Promise<QueryResponse> {
  const trimmed = question.trim();
  if (trimmed.length < 3) {
    throw new Error("Question must be at least 3 characters long.");
  }
  if (trimmed.length > 1000) {
    throw new Error("Question exceeds maximum length of 1000 characters.");
  }

  const endpoint = `${baseUrl}/query`;
  const startTime = Date.now();

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: trimmed }),
    });

    const elapsed = Date.now() - startTime;

    if (!response.ok) {
      let errorDetail = `RAG API request failed with status ${response.status}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          if (Array.isArray(errorJson.detail)) {
            errorDetail = errorJson.detail.map((d: { msg?: string }) => d.msg || "").join(", ");
          } else {
            errorDetail = String(errorJson.detail);
          }
        }
      } catch {
        // Fall back to HTTP status message
      }
      throw new Error(errorDetail);
    }

    const data: QueryResponse = await response.json();

    // Augment metadata with client-measured latency
    return {
      ...data,
      metadata: {
        ...(data.metadata || {}),
        latency_ms: elapsed,
      },
    };
  } catch (error: unknown) {
    if (error instanceof TypeError && error.message.includes("fetch")) {
      throw new Error(
        `Unable to reach RAG backend at ${baseUrl}. Ensure the FastAPI server is running (e.g. uvicorn src.backend_api:app --port 8000).`
      );
    }
    throw error;
  }
}

/**
 * Check backend readiness and liveness.
 *
 * @param baseUrl Optional override for API base URL.
 */
export async function checkHealth(
  baseUrl: string = getApiBaseUrl()
): Promise<HealthStatus> {
  const endpoint = `${baseUrl}/health`;
  const startTime = Date.now();

  const response = await fetch(endpoint, {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  });

  const elapsed = Date.now() - startTime;

  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`);
  }

  const data: HealthStatus = await response.json();
  return { ...data, latency_ms: elapsed };
}

/**
 * Upload and index a document at runtime.
 *
 * @param file Document file (.txt, .md, .pdf)
 * @param baseUrl Optional override for API base URL.
 */
export async function uploadDocument(
  file: File,
  baseUrl: string = getApiBaseUrl()
): Promise<UploadResponse> {
  const endpoint = `${baseUrl}/documents`;
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Document upload failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = String(errorJson.detail);
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}
