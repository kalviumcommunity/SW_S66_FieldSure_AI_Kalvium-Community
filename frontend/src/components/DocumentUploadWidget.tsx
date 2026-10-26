"use client";

import React, { useRef, useState } from "react";
import { uploadDocument } from "../lib/api";
import { UploadResponse } from "../types/rag";

interface DocumentUploadWidgetProps {
  onUploadSuccess?: (res: UploadResponse) => void;
}

export function DocumentUploadWidget({ onUploadSuccess }: DocumentUploadWidgetProps) {
  const [uploading, setUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{
    type: "success" | "error" | "info";
    text: string;
  } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    const validExtensions = [".txt", ".md", ".pdf"];
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    if (!validExtensions.includes(ext)) {
      setStatusMessage({
        type: "error",
        text: `Unsupported file type "${ext}". Supported formats: .txt, .md, .pdf`,
      });
      return;
    }

    setUploading(true);
    setStatusMessage({
      type: "info",
      text: `Ingesting, chunking, and embedding "${file.name}"...`,
    });

    try {
      const response = await uploadDocument(file);
      setStatusMessage({
        type: "success",
        text: `Indexed "${response.filename}": ${response.summary.chunks} chunks (${response.summary.indexed} vectors added). Immediately searchable!`,
      });
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (error: unknown) {
      const msg = error instanceof Error ? error.message : "Upload failed";
      setStatusMessage({
        type: "error",
        text: msg,
      });
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (uploading) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!uploading) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className="widget-card">
      <h3>
        <span>📁 Knowledge Expansion</span>
      </h3>
      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
        Upload clinical protocols or guidelines to index them at runtime without restarting the server.
      </p>

      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md,.pdf"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
          }
        }}
        disabled={uploading}
      />

      <div
        className={`upload-dropzone ${isDragging ? "dragging" : ""}`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Upload document dropzone"
      >
        <span className="upload-icon">📄</span>
        <span className="upload-prompt">
          {uploading ? "Indexing document..." : "Click or drag document here"}
        </span>
        <span className="upload-formats">Supported: PDF, Markdown (.md), Text (.txt)</span>
      </div>

      {statusMessage && (
        <div className={`upload-status-box ${statusMessage.type === "info" ? "uploading" : statusMessage.type}`}>
          {statusMessage.text}
        </div>
      )}
    </div>
  );
}
