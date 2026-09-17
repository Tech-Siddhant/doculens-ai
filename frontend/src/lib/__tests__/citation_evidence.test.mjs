import test from "node:test";
import assert from "node:assert/strict";

import {
  getCitationLabel,
  getEvidenceType,
  getValidationStatus,
  extractBoundingBox,
  parseInlineCitations,
} from "../citationUtils.ts";

test("getCitationLabel formats human-readable labels without raw IDs", () => {
  const tableCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.92,
    document_id: "doc_123",
    page_number: 8,
    metadata: { table_name: "Table 2" },
  };
  assert.equal(getCitationLabel(tableCit), "Page 8 · Table 2");

  const figCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.88,
    document_id: "doc_123",
    page_number: 5,
    metadata: { figure_caption: "Figure 3: System Architecture" },
  };
  assert.equal(getCitationLabel(figCit), "Page 5 · Figure 3: System Archi...");

  const visualCit = {
    reference: "[Evidence 3]",
    rank: 3,
    score: 0.85,
    document_id: "doc_123",
    page_number: 4,
    retrieval_type: "visual",
  };
  assert.equal(getCitationLabel(visualCit), "Page 4 · Visual Page");

  const plainCit = {
    reference: "[Evidence 4]",
    rank: 4,
    score: 0.81,
    document_id: "doc_123",
    page_number: 12,
  };
  assert.equal(getCitationLabel(plainCit), "Page 12");
});

test("getEvidenceType correctly detects modality and classification", () => {
  const textCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.9,
    document_id: "doc_123",
    page_number: 1,
    evidence_text: "Attention mechanisms have become integral.",
  };
  const textType = getEvidenceType(textCit);
  assert.equal(textType.type, "Text Chunk");
  assert.equal(textType.isVisual, false);

  const visualCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.87,
    document_id: "doc_123",
    page_number: 7,
    retrieval_type: "visual",
    image_url: "/api/v1/documents/doc_123/pages/7/image",
  };
  const visualType = getEvidenceType(visualCit);
  assert.equal(visualType.type, "Visual Page");
  assert.equal(visualType.isVisual, true);

  const tableMetaCit = {
    reference: "[Evidence 3]",
    rank: 3,
    score: 0.89,
    document_id: "doc_123",
    page_number: 8,
    metadata: { evidence_type: "table" },
  };
  const tableType = getEvidenceType(tableMetaCit);
  assert.equal(tableType.type, "Table");
  assert.equal(tableType.isVisual, true);
});

test("getValidationStatus accurately reports verified, review, and unverified states", () => {
  // 1. Fully valid grounded citation
  const validCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.95,
    document_id: "doc_123",
    page_number: 3,
    evidence_text: "Sample evidence text",
  };
  assert.equal(getValidationStatus(validCit, true).status, "verified");

  // 2. Ungrounded response notice
  assert.equal(getValidationStatus(validCit, false).status, "review");

  // 3. Invalid citation missing page or document ID
  const invalidCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.1,
    document_id: "",
    page_number: 0,
  };
  assert.equal(getValidationStatus(invalidCit, true).status, "unverified");

  // 4. Explicit rejection in metadata
  const rejectedCit = {
    reference: "[Evidence 3]",
    rank: 3,
    score: 0.5,
    document_id: "doc_123",
    page_number: 4,
    metadata: { validation_status: "rejected" },
  };
  assert.equal(getValidationStatus(rejectedCit, true).status, "unverified");
});

test("extractBoundingBox normalizes bbox coordinates safely", () => {
  // Array format [x0, y0, x1, y1]
  const citArrayBBox = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.9,
    document_id: "doc_123",
    page_number: 2,
    metadata: { bbox: [0.1, 0.2, 0.6, 0.8] },
  };
  const bbox1 = extractBoundingBox(citArrayBBox);
  assert.ok(bbox1);
  assert.equal(bbox1.x, 0.1);
  assert.equal(bbox1.y, 0.2);
  assert.equal(bbox1.width, 0.5);
  assert.equal(bbox1.height, 0.6);
  assert.equal(bbox1.isNormalized, true);

  // Object format { x, y, width, height }
  const citObjBBox = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.9,
    document_id: "doc_123",
    page_number: 2,
    metadata: { bounding_box: { x: 0.05, y: 0.15, width: 0.4, height: 0.3 } },
  };
  const bbox2 = extractBoundingBox(citObjBBox);
  assert.ok(bbox2);
  assert.equal(bbox2.x, 0.05);
  assert.equal(bbox2.y, 0.15);
  assert.equal(bbox2.width, 0.4);
  assert.equal(bbox2.height, 0.3);

  // No bounding box
  const noBBox = {
    reference: "[Evidence 3]",
    rank: 3,
    score: 0.9,
    document_id: "doc_123",
    page_number: 2,
  };
  assert.equal(extractBoundingBox(noBBox), null);
});

test("parseInlineCitations extracts citation tokens and matches Citation objects", () => {
  const citations = [
    {
      reference: "[Evidence 1]",
      rank: 1,
      score: 0.95,
      document_id: "doc_123",
      page_number: 8,
      metadata: { table_name: "Table 2" },
    },
    {
      reference: "[Evidence 2]",
      rank: 2,
      score: 0.9,
      document_id: "doc_123",
      page_number: 5,
    },
  ];

  const answer = "Transformer big achieved 28.4 BLEU [Evidence 1] on English-to-German [2].";
  const segments = parseInlineCitations(answer, citations);

  assert.equal(segments.length, 5);
  assert.equal(segments[0].type, "text");
  assert.equal(segments[0].content, "Transformer big achieved 28.4 BLEU ");

  assert.equal(segments[1].type, "citation");
  assert.equal(segments[1].content, "Page 8 · Table 2");
  assert.equal(segments[1].citation?.page_number, 8);

  assert.equal(segments[3].type, "citation");
  assert.equal(segments[3].content, "Page 5");
  assert.equal(segments[3].citation?.page_number, 5);

  assert.equal(segments[4].type, "text");
  assert.equal(segments[4].content, ".");
});

test("handles missing evidence gracefully", () => {
  const emptyCitations = [];
  const segments = parseInlineCitations("Answer without citations", emptyCitations);
  assert.equal(segments.length, 1);
  assert.equal(segments[0].type, "text");
  assert.equal(segments[0].content, "Answer without citations");

  const ungroundedStatus = getValidationStatus(
    {
      reference: "[Evidence 1]",
      rank: 1,
      score: 0.0,
      document_id: "doc_1",
      page_number: 1,
      evidence_text: "",
    },
    false
  );
  assert.equal(ungroundedStatus.status, "unverified");
});

test("handles invalid citations with missing document ID or invalid page number", () => {
  const noDocCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.85,
    document_id: "",
    page_number: 1,
    evidence_text: "Some text",
  };
  assert.equal(getValidationStatus(noDocCit, true).status, "unverified");

  const zeroPageCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.85,
    document_id: "doc_123",
    page_number: 0,
    evidence_text: "Some text",
  };
  assert.equal(getValidationStatus(zeroPageCit, true).status, "unverified");
});

test("getCitationLabel handles complex section and table names", () => {
  const sectionCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.9,
    document_id: "doc_1",
    page_number: 3,
    metadata: { section_title: "3. Background & Related Work" },
  };
  assert.equal(getCitationLabel(sectionCit), "Page 3 · 3. Background & Relate...");

  const tableCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.91,
    document_id: "doc_1",
    page_number: 7,
    metadata: { table_name: "Table 1" },
  };
  assert.equal(getCitationLabel(tableCit), "Page 7 · Table 1");
});

