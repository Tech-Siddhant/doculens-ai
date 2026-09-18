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

test("Grounded answer flow parses text and inline clickable citations", () => {
  const answer =
    "DocuLens AI uses hybrid retrieval [Evidence 1] and reranks with cross-encoders [Evidence 2] to ensure grounded generation.";
  const citations = [
    {
      reference: "[Evidence 1]",
      rank: 1,
      score: 0.94,
      document_id: "doc_test_123",
      page_number: 12,
      chunk_id: "doc_test_123_p12_c0",
      evidence_text: "DocuLens AI combines dense and BM25 retrievers.",
      retrieval_type: "evidence",
    },
    {
      reference: "[Evidence 2]",
      rank: 2,
      score: 0.89,
      document_id: "doc_test_123",
      page_number: 18,
      chunk_id: "doc_test_123_p18_c1",
      evidence_text: "Cross-encoder scoring ensures high precision.",
      retrieval_type: "evidence",
    },
  ];

  const segments = parseInlineCitations(answer, citations);

  assert.equal(segments.length, 5);
  assert.equal(segments[0].type, "text");
  assert.equal(segments[0].content, "DocuLens AI uses hybrid retrieval ");

  // First citation
  assert.equal(segments[1].type, "citation");
  assert.equal(segments[1].citation?.page_number, 12);
  assert.equal(segments[1].citation?.rank, 1);
  assert.equal(segments[1].content, "Page 12");

  assert.equal(segments[2].type, "text");
  assert.equal(segments[2].content, " and reranks with cross-encoders ");

  // Second citation
  assert.equal(segments[3].type, "citation");
  assert.equal(segments[3].citation?.page_number, 18);
  assert.equal(segments[3].citation?.rank, 2);
  assert.equal(segments[3].content, "Page 18");

  assert.equal(segments[4].type, "text");
  assert.equal(segments[4].content, " to ensure grounded generation.");
});

test("Insufficient evidence response produces ungrounded refusal with no citations", () => {
  const refusalAnswer =
    "I do not have sufficient information in the provided document to answer this question.";
  const citations = [];

  const segments = parseInlineCitations(refusalAnswer, citations);

  assert.equal(segments.length, 1);
  assert.equal(segments[0].type, "text");
  assert.equal(segments[0].content, refusalAnswer);
});

test("Validation status flags ungrounded answers for user review", () => {
  const citation = {
    reference: "[Evidence 1]",
    rank: 1,
    score: 0.75,
    document_id: "doc_test_123",
    page_number: 5,
    evidence_text: "Some evidence excerpt",
  };

  const groundedStatus = getValidationStatus(citation, true);
  assert.equal(groundedStatus.status, "verified");

  const ungroundedStatus = getValidationStatus(citation, false);
  assert.equal(ungroundedStatus.status, "review");
  assert.equal(ungroundedStatus.label, "Needs review");
});

test("Document isolation and invalid citation handling", () => {
  const invalidCitation = {
    reference: "[Evidence 99]",
    rank: 99,
    score: 0.1,
    document_id: "",
    page_number: -1,
  };

  const validation = getValidationStatus(invalidCitation, true);
  assert.equal(validation.status, "unverified");
});

test("Error action mapping for LLM provider timeouts and rate limits", () => {
  const timeout = getRecommendedActionForError(504, "LLM provider timeout");
  assert.equal(timeout.code, "ERR_GATEWAY_TIMEOUT");

  const rateLimit = getRecommendedActionForError(429, "Rate limit exceeded");
  assert.equal(rateLimit.code, "ERR_RATE_LIMIT");

  const unavailable = getRecommendedActionForError(503, "Service unavailable");
  assert.equal(unavailable.code, "ERR_SERVICE_UNAVAILABLE");

  const badGateway = getRecommendedActionForError(502, "Authentication failure");
  assert.equal(badGateway.code, "ERR_BAD_GATEWAY");
});
