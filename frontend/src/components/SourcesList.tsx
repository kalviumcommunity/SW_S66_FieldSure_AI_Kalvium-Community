"use client";

import React from "react";
import { SourceCitation } from "../types/rag";

interface SourcesListProps {
  sources: SourceCitation[];
}

export function SourcesList({ sources }: SourcesListProps) {
  if (!sources || sources.length === 0) {
    return (
      <div className="no-sources-box">
        ℹ️ No sources cited. This indicates either a safe refusal due to insufficient context or an ungrounded response.
      </div>
    );
  }

  return (
    <div className="sources-section">
      <div className="sources-header">
        <h3>
          <span>Retrieved Evidence & Sources</span>
          <span className="sources-count-badge">
            {sources.length} {sources.length === 1 ? "source" : "sources"}
          </span>
        </h3>
      </div>

      <div className="sources-grid">
        {sources.map((src, index) => {
          const scorePercent =
            src.score !== null && src.score !== undefined
              ? Math.round(src.score * 100)
              : null;

          return (
            <div key={`${src.source}-${index}`} className="source-card">
              <div className="source-top">
                <div className="source-name" title={src.source}>
                  📄 {src.source}
                </div>
                {scorePercent !== null && (
                  <div className="source-score" title="Cosine similarity score">
                    {scorePercent}% match
                  </div>
                )}
              </div>

              <div className="source-details">
                {src.chunk_id && (
                  <span>
                    Chunk ID: <span className="chunk-id">{src.chunk_id}</span>
                  </span>
                )}
                <span>Citation Ref: [{index + 1}]</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
