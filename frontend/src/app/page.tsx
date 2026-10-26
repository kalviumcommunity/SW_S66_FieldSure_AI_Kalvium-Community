"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Header } from "../components/Header";
import { QueryInput } from "../components/QueryInput";
import { AnswerCard } from "../components/AnswerCard";
import { DocumentUploadWidget } from "../components/DocumentUploadWidget";
import { SettingsModal } from "../components/SettingsModal";
import { askQuestion, checkHealth, getApiBaseUrl } from "../lib/api";
import { HealthStatus, QueryResponse } from "../types/rag";

export default function Home() {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<QueryResponse | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<string>("");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthLoading, setHealthLoading] = useState<boolean>(true);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [history, setHistory] = useState<
    Array<{ question: string; result: QueryResponse; timestamp: string }>
  >([]);

  // Periodically check backend health
  const refreshHealth = useCallback(async () => {
    setHealthLoading(true);
    setHealthError(null);
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Backend unavailable";
      setHealthError(msg);
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const interval = setInterval(refreshHealth, 30000);
    return () => clearInterval(interval);
  }, [refreshHealth]);

  // Handle RAG question submission
  const handleAskQuestion = async (question: string) => {
    setLoading(true);
    setError(null);
    setCurrentQuestion(question);

    try {
      const result = await askQuestion(question);
      setCurrentResult(result);
      setHistory((prev) => [
        {
          question,
          result,
          timestamp: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 9), // keep last 10
      ]);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Could not get an answer. Please check if the backend service is running and try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header
        health={health}
        healthLoading={healthLoading}
        healthError={healthError}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onRefreshHealth={refreshHealth}
      />

      <div className="content-grid">
        {/* Left Column: Query Workspace */}
        <div className="workspace-card">
          <QueryInput onAsk={handleAskQuestion} loading={loading} />

          {/* Loading Indicator */}
          {loading && (
            <div className="loading-box" role="status" aria-live="polite">
              <div className="spinner" />
              <div className="loading-text">
                <strong>Retrieving evidence & generating grounded answer...</strong>
                <span className="loading-subtext">
                  Checking relevance guardrails, scanning indexed vector chunks, and assembling source citations.
                </span>
              </div>
            </div>
          )}

          {/* Error Banner */}
          {error && !loading && (
            <div className="error-banner" role="alert">
              <div className="error-content">
                <span className="error-icon">⚠️</span>
                <div>
                  <div className="error-title">Query Request Failed</div>
                  <div className="error-message">{error}</div>
                </div>
              </div>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => currentQuestion && handleAskQuestion(currentQuestion)}
                style={{ fontSize: "0.78rem", padding: "6px 12px" }}
              >
                Retry
              </button>
            </div>
          )}

          {/* Grounded Answer & Sources Display */}
          {currentResult && !loading && (
            <div style={{ marginTop: 8 }}>
              <div
                style={{
                  fontSize: "0.85rem",
                  color: "var(--text-muted)",
                  marginBottom: 8,
                }}
              >
                Inquiry: <strong style={{ color: "var(--text-primary)" }}>&ldquo;{currentQuestion}&rdquo;</strong>
              </div>
              <AnswerCard result={currentResult} />
            </div>
          )}
        </div>

        {/* Right Column: Knowledge Expansion & Telemetry */}
        <aside className="sidebar">
          {/* Document Upload Widget */}
          <DocumentUploadWidget
            onUploadSuccess={() => {
              refreshHealth();
            }}
          />

          {/* Query Session History */}
          {history.length > 0 && (
            <div className="widget-card">
              <h3>
                <span>🕒 Recent Inquiries</span>
                <span
                  style={{
                    fontSize: "0.72rem",
                    color: "var(--text-muted)",
                    fontWeight: "normal",
                  }}
                >
                  ({history.length})
                </span>
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {history.map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setCurrentQuestion(item.question);
                      setCurrentResult(item.result);
                      setError(null);
                    }}
                    style={{
                      background: "rgba(15, 23, 42, 0.4)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "var(--radius-sm)",
                      padding: "8px 10px",
                      textAlign: "left",
                      color: "var(--text-secondary)",
                      cursor: "pointer",
                      fontSize: "0.78rem",
                      display: "flex",
                      flexDirection: "column",
                      gap: 4,
                      transition: "all 0.2s ease",
                    }}
                  >
                    <div
                      style={{
                        color: "var(--text-primary)",
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {item.question}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      <span>
                        {item.result.status === "answered" ? "✓ Grounded" : "⚠️ Refused"}
                      </span>
                      <span>{item.timestamp}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* System Telemetry */}
          <div className="widget-card">
            <h3>
              <span>📊 RAG System Diagnostics</span>
            </h3>
            <div>
              <div className="telemetry-row">
                <span>API Endpoint</span>
                <span className="telemetry-value">
                  {typeof window !== "undefined"
                    ? getApiBaseUrl().replace(/^https?:\/\//, "")
                    : "localhost:8001"}
                </span>
              </div>
              <div className="telemetry-row">
                <span>Guardrail Status</span>
                <span className="telemetry-value" style={{ color: "#34d399" }}>
                  Active (Min Score 0.72)
                </span>
              </div>
              <div className="telemetry-row">
                <span>Verification</span>
                <span className="telemetry-value" style={{ color: "#38bdf8" }}>
                  Citation Cross-Check
                </span>
              </div>
              <div className="telemetry-row">
                <span>Vector Index</span>
                <span className="telemetry-value">ChromaDB / In-Memory</span>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onConfigSaved={() => {
          refreshHealth();
        }}
      />
    </div>
  );
}
