"use client";

import React, { useEffect } from "react";
import { DocumentUploadExperience } from "./DocumentUploadExperience";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (documentId?: string) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  // Prevent body scroll and close on Escape while the dialog is open
  useEffect(() => {
    if (!isOpen) return;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = "unset";
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm overflow-y-auto"
      role="dialog"
      aria-modal="true"
      aria-labelledby="upload-modal-title"
      onClick={onClose}
    >
      <div
        className="relative bg-surface border border-border-subtle rounded-xl shadow-2xl w-full max-w-4xl min-h-[min(40rem,90vh)] max-h-[95vh] overflow-y-auto overscroll-contain my-auto m-auto flex flex-col p-6 sm:p-8"
        onClick={(e) => e.stopPropagation()}
      >
        <span id="upload-modal-title" className="sr-only">
          Upload a PDF document
        </span>
        <DocumentUploadExperience
          isModal={true}
          onCloseModal={onClose}
          onUploadSuccess={(docId) => {
            if (onUploadSuccess) onUploadSuccess(docId);
          }}
        />
      </div>
    </div>
  );
};

