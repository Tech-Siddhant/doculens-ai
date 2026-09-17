import React from "react";
import { FileText } from "lucide-react";

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className = "",
}) => {
  return (
    <div
      className={`flex flex-col items-center justify-center p-10 text-center bg-surface-elevated border border-dashed border-border-muted rounded-lg ${className}`}
    >
      <div className="w-12 h-12 rounded-full bg-surface-low flex items-center justify-center text-text-tertiary mb-4">
        {icon || <FileText className="w-6 h-6" />}
      </div>
      <h3 className="font-heading text-base font-semibold text-text-primary mb-1">
        {title}
      </h3>
      <p className="text-sm text-text-secondary max-w-md mb-6">{description}</p>
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
};
