"use client";

import React from "react";
import Link from "next/link";
import { FileText, MessageSquareQuote, Clock } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { DocumentItem } from "@/types";

interface DocumentTableProps {
  documents: DocumentItem[];
  totalCount: number;
}

export const DocumentTable: React.FC<DocumentTableProps> = ({ documents, totalCount }) => {
  return (
    <div className="bg-surface-elevated border border-border-subtle rounded overflow-hidden shadow-xs">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm border-collapse">
          <thead className="bg-surface-low/60 text-[11px] font-semibold text-text-tertiary uppercase tracking-wider border-b border-border-subtle">
            <tr>
              <th scope="col" className="py-3 px-4">Document</th>
              <th scope="col" className="py-3 px-4">Pages</th>
              <th scope="col" className="py-3 px-4">Status</th>
              <th scope="col" className="py-3 px-4">File Size</th>
              <th scope="col" className="py-3 px-4">Uploaded</th>
              <th scope="col" className="py-3 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-subtle">
            {documents.map((doc) => (
              <tr key={doc.id} className="hover:bg-surface-low/50 transition-colors group">
                <td className="py-3.5 px-4">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-red-50 border border-red-100 flex items-center justify-center text-red-600 shrink-0">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 max-w-xs sm:max-w-md">
                      <Link href={`/documents/${doc.id}`} className="font-medium text-text-primary text-sm truncate hover:text-brand transition-colors block">
                        {doc.name}
                      </Link>
                      <p className="font-mono text-[10px] text-text-tertiary mt-0.5">
                        ID: {doc.id}
                      </p>
                    </div>
                  </div>
                </td>
                <td className="py-3.5 px-4 text-text-secondary text-xs">
                  {doc.pageCount > 0 ? (
                    <span>{doc.pageCount} {doc.pageCount === 1 ? "page" : "pages"}</span>
                  ) : (
                    <span className="text-text-tertiary">—</span>
                  )}
                </td>
                <td className="py-3.5 px-4">
                  <StatusBadge status={doc.status} size="sm" />
                </td>
                <td className="py-3.5 px-4 text-xs font-mono text-text-secondary">
                  {doc.fileSizeFormatted}
                </td>
                <td className="py-3.5 px-4 text-xs text-text-tertiary whitespace-nowrap">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="w-3 h-3 text-text-tertiary" />
                    {doc.uploadedAt}
                  </span>
                </td>
                <td className="py-3.5 px-4 text-right whitespace-nowrap">
                  <Link href={`/documents/${doc.id}`}>
                    <Button variant="secondary" size="sm" leftIcon={<MessageSquareQuote className="w-3.5 h-3.5" />} title="Ask question on this document">
                      Ask
                    </Button>
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-3 bg-surface-low/30 border-t border-border-subtle flex items-center justify-between text-xs text-text-tertiary">
        <span>Showing {documents.length} of {totalCount} {totalCount === 1 ? "document" : "documents"}</span>
        <span className="font-mono text-[11px]">DocuLens Multimodal Store</span>
      </div>
    </div>
  );
};
