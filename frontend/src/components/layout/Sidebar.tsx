"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  FileText,
  Settings,
  X,
  FileUp,
} from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const pathname = usePathname();

  const navItems = [
    {
      name: "Home",
      href: "/",
      icon: Home,
      exact: true,
    },
    {
      name: "Documents",
      href: "/documents",
      icon: FileText,
      exact: false,
    },
    {
      name: "Settings",
      href: "/settings",
      icon: Settings,
      exact: false,
    },
  ];

  const isNavActive = (href: string, exact: boolean) => {
    if (exact) {
      return pathname === href;
    }
    return pathname.startsWith(href);
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-xs md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-40 w-60 bg-surface-elevated border-r border-border-subtle flex flex-col transition-transform duration-200 ease-in-out md:translate-x-0 md:static md:z-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Mobile Header inside Sidebar */}
        <div className="flex items-center justify-between h-14 px-4 border-b border-border-subtle md:hidden">
          <span className="font-heading font-semibold text-sm text-text-primary uppercase tracking-wider">
            Navigation
          </span>
          <button
            type="button"
            onClick={onClose}
            className="p-1 text-text-secondary hover:text-text-primary rounded hover:bg-surface-low"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Top Action CTA */}
        <div className="p-3">
          <Link
            href="/documents"
            onClick={() => {
              if (isOpen) onClose();
            }}
            className="flex items-center justify-center gap-2 w-full py-2 px-3 text-xs font-semibold uppercase tracking-wider bg-text-primary text-white rounded hover:bg-brand transition-colors shadow-xs"
          >
            <FileUp className="w-4 h-4" />
            <span>Upload Document</span>
          </Link>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 px-3 py-2 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const active = isNavActive(item.href, item.exact);
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => {
                  if (isOpen) onClose();
                }}
                className={`flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded transition-colors ${
                  active
                    ? "bg-brand-light text-brand font-semibold"
                    : "text-text-secondary hover:text-text-primary hover:bg-surface-low"
                }`}
              >
                <Icon
                  className={`w-4 h-4 shrink-0 ${
                    active ? "text-brand" : "text-text-tertiary"
                  }`}
                />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer info */}
        <div className="p-3 border-t border-border-subtle bg-surface-low/30">
          <div className="px-2 py-1.5 rounded bg-surface-elevated border border-border-subtle text-[11px] text-text-secondary">
            <p className="font-medium text-text-primary">DocuLens AI</p>
            <p className="text-text-tertiary text-[10px] mt-0.5">
              Multimodal Evidence QA
            </p>
          </div>
        </div>
      </aside>
    </>
  );
};
