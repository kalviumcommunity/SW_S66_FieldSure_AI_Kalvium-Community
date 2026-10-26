"use client";

import React from "react";
import { HealthStatus } from "../types/rag";

interface HeaderProps {
  health: HealthStatus | null;
  healthLoading: boolean;
  healthError: string | null;
  onOpenSettings: () => void;
  onRefreshHealth: () => void;
}

export function Header({
  health,
  healthLoading,
  healthError,
  onOpenSettings,
  onRefreshHealth,
}: HeaderProps) {
  let statusClass = "checking";
  let statusText = "Checking API...";

  if (healthLoading) {
    statusClass = "checking";
    statusText = "Checking...";
  } else if (health && health.status === "healthy") {
    statusClass = "online";
    statusText = `Backend Online (${health.latency_ms ?? 0}ms)`;
  } else if (healthError) {
    statusClass = "offline";
    statusText = "Backend Offline";
  }

  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-logo" aria-label="FieldSure AI Logo">
          FS
        </div>
        <div className="brand-text">
          <h1>FieldSure AI</h1>
          <div className="brand-subtitle">
            Grounded Public Health Assistant & RAG Query Inspector
          </div>
        </div>
      </div>

      <div className="header-status">
        <button
          className="status-badge"
          onClick={onRefreshHealth}
          title="Click to re-check API health"
          type="button"
        >
          <span className={`status-dot ${statusClass}`} />
          <span>{statusText}</span>
        </button>

        <button
          className="settings-btn"
          onClick={onOpenSettings}
          title="Configure API Connection"
          type="button"
        >
          ⚙️ Settings
        </button>
      </div>
    </header>
  );
}
