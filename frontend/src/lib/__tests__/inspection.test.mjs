import test from "node:test";
import assert from "node:assert/strict";
import {
  getCitationLabel,
  getEvidenceType,
  getValidationStatus,
  extractBoundingBox,
  parseInlineCitations,
} from "../citationUtils.ts";
import { getRecommendedActionForError } from "../api.ts";

test("Level 1: Citation Label Generation handles text, figures, tables, and fallback", () => {
  const textCit = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.95,
    document_id: "doc_123",
    page_number: 3,
    chunk_id: "chk_1",
    metadata: { section_title: "System Architecture Overview" },
  };
  assert.strictEqual(getCitationLabel(textCit), "Page 3 · System Architecture Ov...");

  const tableCit = {
    reference: "[Evidence 2]",
    rank: 2,
    score: 0.88,
    document_id: "doc_123",
    page_number: 5,
    metadata: { table_name: "Table 1: Benchmark Results" },
  };
  assert.strictEqual(getCitationLabel(tableCit), "Page 5 · Table 1: Benchmark Results");

  const visualCit = {
    reference: "[Evidence 3]",
    rank: 3,
    score: 0.82,
    document_id: "doc_123",
    page_number: 7,
    image_url: "http://example.com/p7.png",
    retrieval_type: "visual",
  };
  assert.strictEqual(getCitationLabel(visualCit), "Page 7 · Visual Page");
});

test("Level 4: Bounding Box and Technical Extraction", () => {
  const citWithBbox = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.9,
    document_id: "doc_1",
    page_number: 2,
    metadata: {
      bbox: [0.1, 0.2, 0.6, 0.8],
    },
  };
  const bbox = extractBoundingBox(citWithBbox);
  assert.ok(bbox);
  assert.strictEqual(bbox.isNormalized, true);
  assert.strictEqual(bbox.x, 0.1);
  assert.strictEqual(bbox.y, 0.2);
  assert.strictEqual(bbox.width, 0.5);
  assert.strictEqual(bbox.height, 0.6);
});

test("Level 1 Inline Citation Parser handles multiple citations and text segments", () => {
  const text = "DocuLens AI provides multimodal retrieval [Evidence 1] with fast cross-encoder reranking [Evidence 2]";
  const citations = [
    { reference: "[Evidence 1]", rank: 1, score: 0.9, document_id: "doc_1", page_number: 1 },
    { reference: "[Evidence 2]", rank: 2, score: 0.85, document_id: "doc_1", page_number: 2 },
  ];
  const parsed = parseInlineCitations(text, citations);
  assert.strictEqual(parsed.length, 4);
  assert.strictEqual(parsed[0].type, "text");
  assert.strictEqual(parsed[1].type, "citation");
  assert.strictEqual(parsed[1].citation?.rank, 1);
  assert.strictEqual(parsed[2].type, "text");
  assert.strictEqual(parsed[3].type, "citation");
  assert.strictEqual(parsed[3].citation?.rank, 2);
});

test("Validation Status evaluates grounded and ungrounded flags accurately", () => {
  const cit = { reference: "[1]", rank: 1, score: 0.8, document_id: "doc_1", page_number: 1, evidence_text: "Sample text" };
  const verified = getValidationStatus(cit, true);
  assert.strictEqual(verified.status, "verified");

  const ungrounded = getValidationStatus(cit, false);
  assert.strictEqual(ungrounded.status, "review");
});

test("Frontend Error Mapping provides clear actionable codes for HTTP 504, 429, 503, 502, 413, and 400", () => {
  const timeoutErr = getRecommendedActionForError(504, "Answer generation timed out after 30 seconds");
  assert.strictEqual(timeoutErr.code, "ERR_GATEWAY_TIMEOUT");
  assert.ok(timeoutErr.recommendation.includes("timed out") || timeoutErr.recommendation.includes("try again"));

  const rateLimitErr = getRecommendedActionForError(429, "Rate limit exceeded");
  assert.strictEqual(rateLimitErr.code, "ERR_RATE_LIMIT");

  const unavailErr = getRecommendedActionForError(503, "Service temporarily unavailable");
  assert.strictEqual(unavailErr.code, "ERR_SERVICE_UNAVAILABLE");

  const badGatewayErr = getRecommendedActionForError(502, "Bad gateway upstream authentication failed");
  assert.strictEqual(badGatewayErr.code, "ERR_BAD_GATEWAY");

  const oversizeErr = getRecommendedActionForError(413, "File exceeds maximum size");
  assert.strictEqual(oversizeErr.code, "ERR_FILE_OVERSIZED");

  const corruptErr = getRecommendedActionForError(400, "Corrupt PDF magic bytes missing");
  assert.strictEqual(corruptErr.code, "ERR_PDF_CORRUPTED_OR_INVALID");
});
