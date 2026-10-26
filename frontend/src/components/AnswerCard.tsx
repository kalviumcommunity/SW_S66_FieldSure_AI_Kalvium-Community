"use client";

import React, { useState } from "react";
import { QueryResponse } from "../types/rag";
import { SourcesList } from "./SourcesList";

interface AnswerCardProps {
  result: QueryResponse;
}

export function AnswerCard({ result }: AnswerCardProps) {
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
    <section className="answer-section">
      <div className="answer-header">
        <div className="answer-title-group">
          <h2>Grounded Answer</h2>
          {isRefusal ? (
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
          {result.metadata?.latency_ms !== undefined && (
            <span className="meta-pill">
              Latency: {result.metadata.latency_ms}ms
            </span>
          )}
        </div>
      </div>

      <div className={`answer-body ${isRefusal ? "refusal" : ""}`}>
        <p>{result.answer}</p>
      </div>

      <div className="answer-actions">
        <button
          type="button"
          className="copy-btn"
          onClick={handleCopy}
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
