"use client";

import React, { useState } from "react";
import { QueryResponse } from "../types/rag";
import { SourcesList } from "./SourcesList";

interface AnswerCardProps {
  result: QueryResponse;
  isStreaming?: boolean;
  isInterrupted?: boolean;
  onRetry?: () => void;
  onStop?: () => void;
}

export function AnswerCard({
  result,
  isStreaming = false,
  isInterrupted = false,
  onRetry,
  onStop,
}: AnswerCardProps) {
  const [copied, setCopied] = useState(false);

  const isRefusal =
    result.status === "refused_weak_context" ||
    result.metadata?.is_refused === true;

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore clipboard error
    }
  };

  const topScore =
    result.metadata?.top_score !== null && result.metadata?.top_score !== undefined
      ? Math.round(result.metadata.top_score * 100)
      : null;

  return (
    <section className="answer-section" aria-live="polite">
      <div className="answer-header">
        <div className="answer-title-group">
          <h2>Grounded Answer</h2>
          {isStreaming ? (
            <span className="badge badge-streaming" title="Tokens are streaming progressively from RAG model">
              <span className="pulse-dot" /> ⚡ Streaming Response...
            </span>
          ) : isInterrupted ? (
            <span className="badge badge-interrupted" title="Stream was interrupted prematurely">
              ⚠️ Incomplete (Stream Stopped)
            </span>
          ) : isRefusal ? (
            <span className="badge badge-refusal" title="Answer was safely refused due to lack of verified context">
              ⚠️ Safe Refusal (Weak Context)
            </span>
          ) : (
            <span className="badge badge-answered" title="Answer is grounded in retrieved clinical documents">
              ✓ Grounded & Verified
            </span>
          )}
        </div>

        <div className="answer-meta-pills">
          {isStreaming && (
            <span className="meta-pill streaming-pill">
              Progressive Stream
            </span>
          )}
          {topScore !== null && (
            <span className="meta-pill" title="Top similarity score">
              Top Match: {topScore}%
            </span>
          )}
          {result.metadata?.chunks_used !== undefined && (
            <span className="meta-pill">
              Chunks: {result.metadata.chunks_used}
            </span>
          )}
          {result.metadata?.latency_ms !== undefined && !isStreaming && (
            <span className="meta-pill">
              Latency: {result.metadata.latency_ms}ms
            </span>
          )}
        </div>
      </div>

      {/* Stream interruption banner */}
      {isInterrupted && (
        <div className="stream-incomplete-notice" role="alert">
          <div>
            <strong>Stream Interrupted:</strong> The answer stopped streaming before completion.
            Partial text and cited sources received prior to interruption are preserved below.
          </div>
          {onRetry && (
            <button
              type="button"
              className="btn-retry-inline"
              onClick={onRetry}
            >
              ↻ Retry Question
            </button>
          )}
        </div>
      )}

      {/* Answer content body */}
      <div className={`answer-body ${isRefusal ? "refusal" : ""} ${isInterrupted ? "interrupted" : ""}`}>
        <p>
          {result.answer || (isStreaming ? "Connecting to stream and preparing context..." : "")}
          {isStreaming && <span className="streaming-cursor" aria-hidden="true">▍</span>}
        </p>
      </div>

      {/* Actions */}
      <div className="answer-actions">
        {isStreaming && onStop && (
          <button
            type="button"
            className="stop-btn"
            onClick={onStop}
            title="Stop generation stream"
          >
            ⏹ Stop Generating
          </button>
        )}
        <button
          type="button"
          className="copy-btn"
          onClick={handleCopy}
          disabled={!result.answer}
          title="Copy answer to clipboard"
        >
          {copied ? "✓ Copied" : "📋 Copy Answer"}
        </button>
      </div>

      {/* Retrieved Sources Section */}
      <SourcesList sources={result.sources} />
    </section>
  );
}
