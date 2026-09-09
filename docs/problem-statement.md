# Problem Statement

# DocuLens AI — Problem Statement

## 1. Project Overview

**DocuLens AI** is a multimodal document intelligence platform designed to answer questions over complex documents using both textual and visual evidence.

The system accepts documents such as research papers and technical PDFs, including documents containing multi-column layouts, tables, figures, charts, scanned pages, and other visually structured content.

Instead of relying exclusively on extracted text, DocuLens AI combines document structure, textual retrieval, visual retrieval, reranking, and LLM/VLM-based generation to produce answers grounded in specific document evidence.

Every generated answer should provide traceable citations, including the relevant document and page whenever possible.

---

## 2. Problem

Conventional document-question-answering systems commonly reduce a PDF to extracted text, split that text into chunks, retrieve similar chunks, and pass them to an LLM.

This approach can lose important information when the meaning of a document depends on:

* tables,
* charts,
* figures,
* page layout,
* scanned content,
* visual relationships,
* captions,
* multi-column structures,
* information distributed across a page.

As a result, a text-only retrieval system may retrieve text that appears semantically relevant while missing the page or visual evidence that actually answers the user's question.

DocuLens AI addresses this limitation by treating a document as both a textual and visual information source.

---

## 3. Target Users

The initial target users are:

* Researchers
* Engineers
* Students
* Technical analysts
* Professionals who regularly work with technical documents

The initial document domain is **research and technical PDFs**.

This domain provides realistic multimodal challenges while keeping the first version sufficiently focused for measurable evaluation.

---

## 4. Core User Problem

A user should be able to upload a technical document and ask questions such as:

* "What methodology did the authors use?"
* "What were the main experimental results?"
* "Which model performed best according to the table?"
* "What does Figure 3 demonstrate?"
* "What limitation did the authors identify?"
* "On which page is this information discussed?"

The system should retrieve the relevant evidence and produce an answer that can be traced back to the source document.

---

## 5. Core Product Workflow

The primary workflow is:

```text
User
  ↓
Upload Document
  ↓
Document Processing
  ↓
Text + Structure + Page Images
  ↓
Indexing
  ↓
User Question
  ↓
Retrieval
  ├── Dense Text Retrieval
  ├── Sparse/BM25 Retrieval
  └── Visual Retrieval
  ↓
Hybrid Fusion
  ↓
Reranking
  ↓
Evidence Validation
  ↓
LLM/VLM Generation
  ↓
Grounded Answer
  ↓
Page-Level Citations + Evidence
```

The initial implementation will deliberately begin with a text-only retrieval baseline before multimodal retrieval is introduced.

This allows the project to measure whether each additional retrieval capability actually improves system performance.

---

## 6. Why Multimodal Retrieval?

The project is not intended to be another generic "chat with PDF" application.

The central technical problem is retrieving evidence from documents where relevant information may be represented through both language and visual structure.

The system therefore needs to support multiple evidence channels:

### Text evidence

Useful for:

* paragraphs,
* headings,
* explanations,
* definitions,
* textual descriptions.

### Visual evidence

Useful for:

* charts,
* tables,
* figures,
* scanned pages,
* layout-dependent information,
* visually structured pages.

### Hybrid evidence

Some questions may require both.

For example, a question may require a textual explanation combined with a result presented in a table or figure.

---

## 7. Product Scope

### MVP

The MVP will implement:

1. PDF upload.
2. PDF validation.
3. Text extraction.
4. Metadata extraction.
5. Text chunking.
6. Dense text embeddings.
7. Vector storage.
8. Top-k retrieval.
9. LLM-based answer generation.
10. Source/page citations.
11. Basic API.
12. Basic tests.

The MVP establishes the text-only baseline.

---

### V1

V1 will extend the baseline with:

1. Page rendering.
2. Structured document extraction.
3. Page-level visual representations.
4. Visual retrieval.
5. Dense text retrieval.
6. BM25 retrieval.
7. Hybrid retrieval.
8. Retrieval score fusion.
9. Cross-encoder reranking.
10. Evidence validation.
11. Grounded LLM/VLM generation.
12. Page previews.
13. Citation-aware responses.
14. Evaluation dataset.
15. Retrieval evaluation.
16. Generation evaluation.
17. Web frontend.
18. Docker-based local deployment.
19. Observability and structured logging.
20. User-provided AI provider credentials.

---

## 8. Advanced Scope

Potential later capabilities include:

* table extraction,
* bounding-box evidence highlighting,
* document comparison,
* OCR fallback,
* user feedback loops,
* caching,
* multi-tenant indexing,
* authentication and authorization improvements,
* richer observability,
* cloud deployment,
* more sophisticated agentic workflows.

These capabilities will only be added when there is a demonstrated product or engineering requirement.

---

## 9. Agentic AI Strategy

DocuLens AI will use agents selectively.

Agents will be used for bounded reasoning and orchestration rather than replacing deterministic components unnecessarily.

Potential agent responsibilities include:

* query analysis,
* retrieval strategy selection,
* tool orchestration,
* evidence validation,
* controlled retry/fallback decisions.

Retrieval systems, document parsing, storage, validation, and other deterministic operations should remain explicit services or tools where deterministic behavior is preferable.

The agent must operate within defined tools, permissions, schemas, and execution limits.

The system should not provide an autonomous agent with unrestricted access to application resources.

---

## 10. Bring Your Own Key (BYOK)

DocuLens AI will follow a **Bring Your Own Key** model for external AI inference.

Users provide their own supported AI provider credentials and select the model they want to use.

DocuLens AI will not provide unlimited access to externally hosted AI models using a shared project-owned API key.

Provider credentials must be handled as secrets.

The system must:

* avoid exposing credentials to the browser unnecessarily,
* never log raw API keys,
* mask credentials in the UI,
* validate credentials before use where supported,
* isolate credentials between users,
* support credential revocation,
* prevent credentials from appearing in application errors or traces.

The exact credential-storage strategy will be finalized during the security architecture stage.

---

## 11. Success Metrics

The project will evaluate the system at both retrieval and generation levels.

### Retrieval metrics

* Recall@K
* MRR@K
* retrieval latency

### Generation metrics

* Faithfulness
* Answer relevancy
* Context precision
* Context recall

### System metrics

* document ingestion time,
* pages processed,
* chunks created,
* retrieval latency,
* generation latency,
* token usage,
* model used,
* failure rate.

The project will compare the baseline against progressively more capable retrieval configurations.

Example:

```text
Text-only baseline
        ↓
Dense retrieval
        ↓
Dense + BM25
        ↓
Dense + BM25 + Visual
        ↓
Hybrid + Reranking
```

No performance improvement will be claimed until it is experimentally measured.

---

## 12. Evidence and Grounding Requirements

A successful response should not simply produce an answer.

It should provide evidence that allows the user to verify the answer.

The target response structure is conceptually:

```json
{
  "answer": "...",
  "citations": [
    {
      "document_id": "...",
      "page": 5,
      "evidence_text": "..."
    }
  ]
}
```

If the system cannot retrieve sufficient evidence, it should avoid presenting unsupported information as fact.

The system should prefer:

```text
"I could not find sufficient evidence in the provided documents."
```

over generating an unsupported answer.

---

## 13. Security Considerations

The system will treat uploaded documents and provider credentials as potentially sensitive.

Relevant security requirements include:

* file-size limits,
* supported-file validation,
* safe document processing,
* secret protection,
* API authentication,
* authorization,
* isolation of user documents,
* protection against prompt injection,
* controlled tool access,
* rate limiting,
* request validation,
* safe error handling,
* sensitive-data exclusion from logs.

Security controls will be introduced according to the actual capabilities of each implementation stage rather than being added as unused abstractions.

---

## 14. Reliability Requirements

The system should explicitly handle:

* invalid PDFs,
* unsupported documents,
* corrupted files,
* extraction failures,
* page-rendering failures,
* empty retrieval,
* poor retrieval,
* model timeouts,
* provider failures,
* malformed model responses,
* unsupported questions,
* insufficient evidence,
* excessive document size,
* API quota failures.

Fallback behavior should be explicit and observable.

---

## 15. Non-Goals for the Initial Implementation

The initial implementation will not attempt to build:

* a foundation model,
* custom model pretraining,
* unnecessary microservices,
* Kubernetes infrastructure,
* distributed GPU infrastructure,
* unrestricted autonomous agents,
* fine-tuning without an identified requirement,
* multiple databases without a demonstrated need,
* enterprise-scale claims without load and reliability evidence.

---

## 16. Portfolio Objective

The goal is not simply to demonstrate that an LLM can answer questions about a PDF.

The project should demonstrate the engineering progression:

```text
Baseline
   ↓
Measurement
   ↓
Improved Retrieval
   ↓
Multimodal Retrieval
   ↓
Reranking
   ↓
Grounded Generation
   ↓
Agentic Orchestration
   ↓
Evaluation
   ↓
Security
   ↓
Observability
   ↓
Deployment
```

The strongest evidence will come from experiments showing how each architectural improvement affects retrieval quality, answer quality, latency, and cost.

---

## 17. Definition of Success

DocuLens AI will be considered successful when a user can:

1. Open the web application.
2. Configure their own supported AI provider.
3. Upload a technical PDF.
4. Wait for processing to complete.
5. Ask a natural-language question.
6. Receive a grounded answer.
7. Inspect the supporting evidence.
8. Navigate to the cited page.
9. Understand which retrieval path contributed to the result.
10. Receive an explicit failure response when sufficient evidence cannot be found.

From an engineering perspective, success additionally requires that the system's retrieval and generation quality can be measured against a defined evaluation dataset.

---

## 18. Guiding Engineering Principle

> **Build the simplest system that can prove the next technical hypothesis.**

Every major component must have a measurable purpose.

A technology should be added because it solves a demonstrated problem—not because it is fashionable or impressive on a technology list.
