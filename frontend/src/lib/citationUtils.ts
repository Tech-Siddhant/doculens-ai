import type { Citation } from "../types";

export type ValidationStatus = "verified" | "review" | "unverified";

export interface ValidationInfo {
  status: ValidationStatus;
  label: string;
  description: string;
}

export interface BoundingBoxRect {
  x: number;
  y: number;
  width: number;
  height: number;
  isNormalized: boolean;
}

export interface ParsedTextSegment {
  type: "text" | "citation";
  content: string;
  citation?: Citation;
  index?: number;
}

/**
 * Returns a clean, human-readable citation label (e.g., "Page 8 · Table 2" or "Page 8").
 * Avoids exposing raw internal IDs.
 */
export function getCitationLabel(citation: Citation): string {
  const page = citation.page_number > 0 ? `Page ${citation.page_number}` : "Source";
  const meta = citation.metadata || {};

  if (typeof meta.table_name === "string" && meta.table_name.trim()) {
    return `${page} · ${meta.table_name.trim()}`;
  }
  if (typeof meta.figure_caption === "string" && meta.figure_caption.trim()) {
    const caption = meta.figure_caption.trim();
    const shortCap = caption.length > 25 ? `${caption.slice(0, 22)}...` : caption;
    return `${page} · ${shortCap}`;
  }
  if (typeof meta.section_title === "string" && meta.section_title.trim()) {
    const section = meta.section_title.trim();
    const shortSec = section.length > 25 ? `${section.slice(0, 22)}...` : section;
    return `${page} · ${shortSec}`;
  }
  if (typeof meta.evidence_type === "string" && meta.evidence_type.trim()) {
    const typeName = meta.evidence_type.trim();
    const capType = typeName.charAt(0).toUpperCase() + typeName.slice(1);
    return `${page} · ${capType}`;
  }
  if (citation.retrieval_type === "visual" || Boolean(citation.image_url)) {
    return `${page} · Visual Page`;
  }
  if (citation.reference && !citation.reference.toLowerCase().includes("evidence")) {
    return `${page} · ${citation.reference}`;
  }
  return page;
}

/**
 * Returns the human-readable evidence classification and visual indicator.
 */
export function getEvidenceType(citation: Citation): {
  type: string;
  isVisual: boolean;
} {
  const meta = citation.metadata || {};
  if (typeof meta.evidence_type === "string" && meta.evidence_type.trim()) {
    const raw = meta.evidence_type.trim().toLowerCase();
    if (raw.includes("table")) return { type: "Table", isVisual: true };
    if (raw.includes("figure")) return { type: "Figure", isVisual: true };
    if (raw.includes("chart")) return { type: "Chart", isVisual: true };
    if (raw.includes("image")) return { type: "Image", isVisual: true };
    return {
      type: raw.charAt(0).toUpperCase() + raw.slice(1),
      isVisual: Boolean(citation.image_url) || citation.retrieval_type === "visual",
    };
  }

  if (meta.table_name) return { type: "Table", isVisual: true };
  if (meta.figure_caption) return { type: "Figure", isVisual: true };

  const isVisual =
    citation.retrieval_type === "visual" || Boolean(citation.image_url);

  return {
    type: isVisual ? "Visual Page" : "Text Chunk",
    isVisual,
  };
}

/**
 * Derives accurate validation status without fabricating verification.
 */
export function getValidationStatus(
  citation: Citation,
  isGrounded: boolean = true
): ValidationInfo {
  const meta = citation.metadata || {};
  const explicitStatus =
    typeof meta.validation_status === "string"
      ? meta.validation_status.toLowerCase()
      : null;

  if (explicitStatus === "verified" || explicitStatus === "valid") {
    return {
      status: "verified",
      label: "Verified",
      description: "Grounded against verified document content",
    };
  }
  if (
    explicitStatus === "review" ||
    explicitStatus === "needs_review" ||
    explicitStatus === "warning"
  ) {
    return {
      status: "review",
      label: "Needs review",
      description: "Partial or unverified evidence match",
    };
  }
  if (
    explicitStatus === "unverified" ||
    explicitStatus === "invalid" ||
    explicitStatus === "rejected"
  ) {
    return {
      status: "unverified",
      label: "Not verified",
      description: "Citation could not be verified against document",
    };
  }

  // Schema-level validation check
  const hasValidPage = citation.page_number > 0;
  const hasDocId = Boolean(citation.document_id && citation.document_id.trim());
  const hasContent = Boolean(
    (citation.evidence_text && citation.evidence_text.trim()) ||
      (citation.text && citation.text.trim()) ||
      citation.image_url
  );

  if (!hasValidPage || !hasDocId || !hasContent) {
    return {
      status: "unverified",
      label: "Not verified",
      description: "Incomplete citation or missing source reference",
    };
  }

  if (!isGrounded) {
    return {
      status: "review",
      label: "Needs review",
      description: "Retrieved evidence was not definitively grounded in generation",
    };
  }

  return {
    status: "verified",
    label: "Verified",
    description: "Grounded against document content",
  };
}

/**
 * Safely extracts normalized bounding box coordinates if provided in metadata.
 * Does not fabricate bounding boxes.
 */
export function extractBoundingBox(
  citation: Citation
): BoundingBoxRect | null {
  const meta = citation.metadata || {};
  const bbox = meta.bbox || meta.bounding_box;

  if (Array.isArray(bbox) && bbox.length === 4) {
    const [x0, y0, x1, y1] = bbox.map(Number);
    if (!isNaN(x0) && !isNaN(y0) && !isNaN(x1) && !isNaN(y1)) {
      const isNorm = x0 <= 1 && y0 <= 1 && x1 <= 1 && y1 <= 1;
      const minX = Math.min(x0, x1);
      const minY = Math.min(y0, y1);
      const w = Math.abs(x1 - x0);
      const h = Math.abs(y1 - y0);
      return {
        x: Math.round(minX * 10000) / 10000,
        y: Math.round(minY * 10000) / 10000,
        width: Math.round(w * 10000) / 10000,
        height: Math.round(h * 10000) / 10000,
        isNormalized: isNorm,
      };
    }
  }

  if (
    bbox &&
    typeof bbox === "object" &&
    "x" in bbox &&
    "y" in bbox &&
    "width" in bbox &&
    "height" in bbox
  ) {
    const b = bbox as { x: number; y: number; width: number; height: number };
    const x = Number(b.x);
    const y = Number(b.y);
    const w = Number(b.width);
    const h = Number(b.height);
    if (!isNaN(x) && !isNaN(y) && !isNaN(w) && !isNaN(h)) {
      const isNorm = x <= 1 && y <= 1 && w <= 1 && h <= 1;
      return {
        x: Math.round(x * 10000) / 10000,
        y: Math.round(y * 10000) / 10000,
        width: Math.round(w * 10000) / 10000,
        height: Math.round(h * 10000) / 10000,
        isNormalized: isNorm,
      };
    }
  }

  return null;
}

/**
 * Parses answer text with inline citation tags (e.g. "[Evidence 1]", "[1]", "[1, 2]")
 * into structured text and clickable citation nodes.
 */
export function parseInlineCitations(
  text: string,
  citations: Citation[]
): ParsedTextSegment[] {
  if (!text) return [];
  if (!citations || citations.length === 0) {
    return [{ type: "text", content: text }];
  }

  const pattern = /\[\s*(?:Evidence\s*:?\s*)?(\d+(?:\s*,\s*(?:Evidence\s*:?\s*)?\d+)*)\s*\]/gi;
  const segments: ParsedTextSegment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({
        type: "text",
        content: text.slice(lastIndex, match.index),
      });
    }

    const rawIndicesStr = match[1];
    const extractedNumbers = rawIndicesStr
      .replace(/Evidence\s*:?/gi, "")
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n));

    for (const num of extractedNumbers) {
      const matched =
        citations.find((c) => c.rank === num) ||
        citations[num - 1] ||
        citations.find((c) => c.page_number === num);

      segments.push({
        type: "citation",
        content: matched ? getCitationLabel(matched) : `[${num}]`,
        citation: matched,
        index: num,
      });
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < text.length) {
    segments.push({
      type: "text",
      content: text.slice(lastIndex),
    });
  }

  return segments;
}
