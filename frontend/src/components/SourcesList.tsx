"use client";

import React, { useState } from "react";
import { SourceCitation } from "../types/rag";

interface SourcesListProps {
  sources: SourceCitation[];
}

export function SourcesList({ sources }: SourcesListProps) {
  const [allOpen, setAllOpen] = useState<boolean | null>(null);

  if (!sources || sources.length === 0) {
    return (
      <div className="no-sources-box" role="status">
        ℹ️ No sources cited. This indicates either a safe refusal due to insufficient context or an ungrounded response.
      </div>
    );
  }

  const toggleAll = () => {
    setAllOpen((prev) => (prev === true ? false : true));
  };

  return (
    <section aria-label="Sources" className="sources-section">
      <div className="sources-header">
        <h3>
          <span>📚 Cited Evidence & Sources</span>
          <span className="sources-count-badge">
            {sources.length} {sources.length === 1 ? "source" : "sources"}
          </span>
        </h3>
        <button
          type="button"
          className="btn-toggle-all"
          onClick={toggleAll}
          title={allOpen ? "Collapse all source chunks" : "Expand all source chunks"}
        >
          {allOpen ? "Collapse All" : "Expand All"}
        </button>
      </div>

      <div className="sources-list-container">
        {sources.map((source, index) => {
          const marker = source.label || `[${index + 1}]`;
          const docName = source.document || source.source || "Official Guidance Document";
          const chunkId = source.chunk_id || null;
          const scorePercent =
            source.score !== null && source.score !== undefined
              ? Math.round(source.score * 100)
              : null;
          const sourceId = source.id || `source-${index + 1}`;

          return (
            <details
              key={sourceId}
              className="citation-details"
              open={allOpen !== null ? allOpen : undefined}
            >
              <summary className="citation-summary">
                <div className="summary-left">
                  <span className="citation-marker">{marker}</span>
                  <span className="citation-document" title={docName}>
                    📄 {docName}
                  </span>
                  {chunkId && (
                    <span className="citation-chunk-badge" title={`Chunk ID: ${chunkId}`}>
                      {chunkId}
                    </span>
                  )}
                  {source.section && (
                    <span className="citation-section-badge" title={`Section: ${source.section}`}>
                      § {source.section}
                    </span>
                  )}
                </div>

                <div className="summary-right">
                  {scorePercent !== null && (
                    <span className="source-score" title="Cosine similarity match score">
                      {scorePercent}% match
                    </span>
                  )}
                  <span className="expand-indicator" aria-hidden="true">
                    ▾
                  </span>
                </div>
              </summary>

              <div className="citation-body">
                <div className="citation-text-header">
                  <span className="citation-text-title">Retrieved Source Passage:</span>
                  {source.text && (
                    <button
                      type="button"
                      className="copy-source-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigator.clipboard.writeText(source.text || "");
                      }}
                      title="Copy cited source text"
                    >
                      📋 Copy Text
                    </button>
                  )}
                </div>

                <p className="citation-text">
                  {source.text || "No detailed passage text was returned with this source record."}
                </p>
              </div>
            </details>
          );
        })}
      </div>
    </section>
  );
}
