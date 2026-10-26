"use client";

import React, { useState } from "react";

interface QueryInputProps {
  onAsk: (question: string) => void;
  loading: boolean;
}

const SAMPLE_QUESTIONS = [
  "What evidence is required for project submission?",
  "What are the core immunization guidelines?",
  "What is the recommended dosage for pediatric malaria treatment?",
  "Can you prescribe medications outside the approved formulary?",
];

export function QueryInput({ onAsk, loading }: QueryInputProps) {
  const [question, setQuestion] = useState("");

  const trimmed = question.trim();
  const charCount = question.length;
  const isValid = trimmed.length >= 3 && trimmed.length <= 1000;

  let counterClass = "char-counter";
  if (charCount > 900 && charCount <= 1000) {
    counterClass = "char-counter limit-near";
  } else if (charCount > 1000) {
    counterClass = "char-counter limit-exceeded";
  }

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!isValid || loading) return;
    onAsk(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSelectSample = (sample: string) => {
    setQuestion(sample);
  };

  const handleClear = () => {
    setQuestion("");
  };

  return (
    <div className="query-box">
      <div className="query-label">
        <span>Clinical or Operational Inquiry</span>
        <span className={counterClass}>
          {charCount} / 1000 chars {charCount > 0 && charCount < 3 && "(min 3)"}
        </span>
      </div>

      <form onSubmit={handleSubmit} className="input-container">
        <textarea
          className="query-textarea"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about clinical guidelines, submission evidence, or malaria protocols... (Press Enter to submit)"
          disabled={loading}
          rows={3}
        />

        <div className="input-actions">
          <div className="input-tips">
            Press <kbd>Enter</kbd> to submit, <kbd>Shift + Enter</kbd> for newline
          </div>

          <div className="button-group">
            {question && (
              <button
                type="button"
                className="btn-secondary"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </button>
            )}

            <button
              type="submit"
              className="btn-primary"
              disabled={!isValid || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  Processing Query...
                </>
              ) : (
                <>Ask Assistant →</>
              )}
            </button>
          </div>
        </div>
      </form>

      <div className="samples-wrapper">
        <div className="samples-title">Quick Test Prompts</div>
        <div className="samples-list">
          {SAMPLE_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              type="button"
              className="sample-pill"
              onClick={() => handleSelectSample(q)}
              disabled={loading}
              title={q}
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
