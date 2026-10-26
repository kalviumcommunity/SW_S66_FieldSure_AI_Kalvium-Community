"use client";

import React, { useState, useEffect } from "react";
import { getApiBaseUrl, setApiBaseUrl, DEFAULT_API_URL, checkHealth } from "../lib/api";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfigSaved: () => void;
}

export function SettingsModal({ isOpen, onClose, onConfigSaved }: SettingsModalProps) {
  const [url, setUrl] = useState(DEFAULT_API_URL);
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setUrl(getApiBaseUrl());
      setTestResult(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await checkHealth(url);
      setTestResult(`✓ Connected to ${res.service} v${res.version} (${res.latency_ms}ms)`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Connection failed";
      setTestResult(`✗ Error: ${msg}`);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    setApiBaseUrl(url);
    onConfigSaved();
    onClose();
  };

  const handleReset = () => {
    setUrl(DEFAULT_API_URL);
    setApiBaseUrl(DEFAULT_API_URL);
    setTestResult(null);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>⚙️ Backend Connection Settings</h3>
          <button className="close-btn" onClick={onClose} type="button">
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="modal-field">
            <label htmlFor="api-url-input">RAG Backend API URL</label>
            <input
              id="api-url-input"
              className="modal-input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://localhost:8001"
            />
            <span style={{ fontSize: "0.74rem", color: "var(--text-muted)" }}>
              The FastAPI endpoint handling <code>/query</code>, <code>/documents</code>, and <code>/health</code>.
            </span>
          </div>

          {testResult && (
            <div
              style={{
                fontSize: "0.8rem",
                padding: "8px 12px",
                borderRadius: "var(--radius-sm)",
                background: testResult.startsWith("✓")
                  ? "var(--accent-emerald-subtle)"
                  : "var(--accent-rose-subtle)",
                color: testResult.startsWith("✓") ? "#6ee7b7" : "#fda4af",
              }}
            >
              {testResult}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button
            type="button"
            className="btn-secondary"
            onClick={handleTest}
            disabled={testing}
          >
            {testing ? "Testing..." : "Test Connection"}
          </button>

          <button
            type="button"
            className="btn-secondary"
            onClick={handleReset}
          >
            Reset Default
          </button>

          <button
            type="button"
            className="btn-primary"
            onClick={handleSave}
          >
            Save & Apply
          </button>
        </div>
      </div>
    </div>
  );
}
