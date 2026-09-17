"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Menu, Layers, UploadCloud } from "lucide-react";
import { apiClient } from "@/lib/api";
import { UploadModal } from "@/components/documents/UploadModal";

interface HeaderProps {
  onMenuToggle?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onMenuToggle }) => {
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [backendVersion, setBackendVersion] = useState<string>("");
  const [isUploadOpen, setIsUploadOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;
    apiClient
      .checkHealth()
      .then((res) => {
        if (isMounted) {
          setBackendStatus("online");
          setBackendVersion(res.version);
        }
      })
      .catch(() => {
        if (isMounted) {
          setBackendStatus("offline");
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-14 px-4 sm:px-6 bg-surface-elevated border-b border-border-subtle">
      <div className="flex items-center gap-3">
        {/* Mobile menu trigger */}
        <button
          type="button"
          onClick={onMenuToggle}
          className="p-1.5 -ml-1.5 text-text-secondary hover:text-text-primary rounded md:hidden hover:bg-surface-low transition-colors"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Brand Logo & Name */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="w-8 h-8 rounded bg-text-primary flex items-center justify-center text-white shadow-xs group-hover:bg-brand transition-colors">
            <Layers className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-heading font-bold text-base tracking-tight text-text-primary">
              DOCULENS<span className="text-brand">AI</span>
            </span>
            <span className="hidden sm:inline-block font-mono text-[10px] text-text-tertiary uppercase tracking-wider font-semibold">
              RESEARCH
            </span>
          </div>
        </Link>
      </div>

      {/* Right Header Status / Meta */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setIsUploadOpen(true)}
          className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded bg-brand text-white hover:bg-brand-hover transition-colors"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Upload</span>
        </button>

        <div
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded border border-border-subtle bg-surface-low"
          title={`Backend status: ${backendStatus}`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              backendStatus === "online"
                ? "bg-status-ready"
                : backendStatus === "offline"
                ? "bg-status-failed"
                : "bg-status-processing animate-pulse"
            }`}
          />
          <span className="font-mono text-[11px] text-text-secondary">
            {backendStatus === "online"
              ? `API v${backendVersion || "0.1.0"}`
              : backendStatus === "offline"
              ? "API Offline"
              : "Checking API..."}
          </span>
        </div>
      </div>

      <UploadModal 
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onUploadSuccess={() => {}}
      />
    </header>
  );
};
