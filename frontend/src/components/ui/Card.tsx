import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  header?: React.ReactNode;
  footer?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = "",
  header,
  footer,
}) => {
  return (
    <div
      className={`bg-surface-elevated border border-border-subtle rounded shadow-sm overflow-hidden ${className}`}
    >
      {header && (
        <div className="px-5 py-4 border-b border-border-subtle bg-surface-elevated">
          {header}
        </div>
      )}
      <div className="p-5">{children}</div>
      {footer && (
        <div className="px-5 py-3.5 border-t border-border-subtle bg-surface-low/50">
          {footer}
        </div>
      )}
    </div>
  );
};
