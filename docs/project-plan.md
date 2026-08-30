 

````markdown
# DocuLens AI — Project Plan

## 1. Project Objective

DocuLens AI is a multimodal document intelligence system for answering questions over research and technical PDFs.

The project will evolve from a text-only RAG baseline into a multimodal retrieval system that can retrieve textual and visual evidence and generate grounded answers with page-level citations.

The core engineering hypothesis is:

> Multimodal evidence retrieval can improve document question-answering performance on visually structured PDFs compared with a text-only retrieval baseline.

All improvements must be experimentally measured before being claimed.

---

## 2. Project Roadmap

```text
Phase 1
Repository & Development Foundation
        ↓
Phase 2
Text RAG Baseline
        ↓
Phase 3
Visual Document Pipeline
        ↓
Phase 4
Hybrid Retrieval
        ↓
Phase 5
Reranking & Grounded Generation
        ↓
Phase 6
Evaluation & Failure Analysis
        ↓
Phase 7
Frontend Product Demo
        ↓
Phase 8
Reproducible Deployment
        ↓
Phase 9
Reliability & Observability
        ↓
Phase 10
Security & BYOK
        ↓
Phase 11
Bounded Agentic Orchestration
        ↓
Phase 12
Advanced Capabilities
````

---

## 3. Phase Summary

### Phase 1 — Repository & Development Foundation

**Objective:**
Create a clean, runnable development foundation.

**Deliverables:**

* Repository structure
* Python project configuration
* FastAPI application
* Configuration management
* Health endpoint
* Basic testing setup
* Environment template
* Initial README

**Gate:**
Application runs locally and tests execute successfully.

---

### Phase 2 — Text RAG Baseline

**Objective:**
Build the simplest complete document QA system and establish the baseline.

**Flow:**

```text
PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ Vector Retrieval
→ LLM
→ Answer + Citation
```

**Deliverables:**

* PDF ingestion
* Text extraction
* Metadata
* Chunking
* Dense embeddings
* Vector retrieval
* Basic answer generation
* Page/source citations

**Gate:**
A technical PDF can be ingested and queried end-to-end.

---

### Phase 3 — Visual Document Pipeline

**Objective:**
Add page-level visual representations for information that text extraction may lose.

**Deliverables:**

* PDF page rendering
* Page metadata
* Visual representations
* Visual retrieval
* Page-level evidence mapping

**Gate:**
Relevant visual pages can be retrieved for appropriate queries.

---

### Phase 4 — Hybrid Retrieval

**Objective:**
Combine complementary retrieval methods.

**Retrieval channels:**

```text
Dense Retrieval
+
BM25
+
Visual Retrieval
```

**Deliverables:**

* BM25 retrieval
* Retrieval score/rank handling
* Fusion
* Candidate merging
* Retrieval comparison experiments

**Gate:**
Hybrid retrieval can be compared against the existing baseline using retrieval metrics.

---

### Phase 5 — Reranking & Grounded Generation

**Objective:**
Improve evidence selection and constrain generation to retrieved evidence.

**Deliverables:**

* Cross-encoder reranking
* Context assembly
* Grounding logic
* Citation validation
* Structured responses
* Insufficient-evidence handling

**Gate:**
Answers are traceable to retrieved document evidence and unsupported answers are handled explicitly.

---

### Phase 6 — Evaluation & Failure Analysis

**Objective:**
Measure system quality and determine whether architectural improvements actually help.

**Deliverables:**

* Evaluation dataset
* Retrieval evaluation
* Generation evaluation
* Baseline comparisons
* Failure categorization
* Experiment results

**Primary metrics:**

```text
Recall@K
MRR@K
Faithfulness
Answer Relevancy
Context Precision
Context Recall
Latency
```

**Gate:**
Experiments are reproducible and measured results are documented.

---

### Phase 7 — Frontend Product Demo

**Objective:**
Provide a simple interface for demonstrating the complete workflow.

**Deliverables:**

* Document upload
* Document library
* Question interface
* Answer display
* Citation/page evidence viewer

**Gate:**
A user can complete the core workflow through the UI.

---

### Phase 8 — Reproducible Deployment

**Objective:**
Make the project easy to run in a clean environment.

**Deliverables:**

* Docker configuration
* Local service setup
* Environment configuration
* Reproducible setup instructions

**Gate:**
Another developer can run the system using the documented setup.

---

### Phase 9 — Reliability & Observability

**Objective:**
Make important system behavior and failures visible.

**Focus:**

* Processing failures
* Retrieval failures
* Model failures
* Timeouts
* Latency
* Token usage
* Retrieval diagnostics
* Structured logging

**Gate:**
Important failure paths are handled and observable.

---

### Phase 10 — Security & BYOK

**Objective:**
Secure uploaded documents and external AI provider credentials.

**Focus:**

* Credential protection
* Provider configuration
* User/document isolation
* File validation
* Rate limiting
* Prompt-injection considerations
* Safe error handling

**Gate:**
Security-sensitive flows are documented and tested.

---

### Phase 11 — Bounded Agentic Orchestration

**Objective:**
Introduce agents only if they provide measurable value.

**Potential responsibilities:**

* Query analysis
* Retrieval strategy selection
* Tool orchestration
* Evidence validation
* Controlled retries

**Gate:**
Agentic orchestration demonstrates measurable benefit over the deterministic workflow.

---

### Phase 12 — Advanced Capabilities

Only build capabilities justified by actual requirements or evaluation results.

Potential examples:

* OCR fallback
* Table extraction
* Bounding-box evidence
* Document comparison
* Feedback loops
* Caching
* Multi-document reasoning
* Cloud deployment
* Scaling improvements

---

## 4. Dependencies

```text
Phase 1
  ↓
Phase 2
  ↓
Phase 3
  ↓
Phase 4
  ↓
Phase 5
  ↓
Phase 6
```

Frontend, deployment, observability, and security can build on the validated core system.

Agentic orchestration comes after the underlying retrieval and grounding capabilities are independently functional.

---

## 5. Project Gates

The project progresses based on evidence, not simply completed code.

### Baseline Gate

Text RAG works and can be measured.

### Multimodal Gate

Visual retrieval works and page mapping is preserved.

### Retrieval Gate

Hybrid retrieval can be compared experimentally.

### Grounding Gate

Answers and citations can be validated.

### Evaluation Gate

A reproducible evaluation process exists.

### Product Gate

The core workflow works through the frontend.

### Deployment Gate

The system can be reproduced locally.

### Hardening Gate

Important reliability and security risks are addressed.

---

## 6. Out of Scope Initially

Do not build initially:

* Foundation-model training
* Custom pretraining
* Fine-tuning without demonstrated need
* Kubernetes
* Distributed GPU infrastructure
* Unnecessary microservices
* Unrestricted autonomous agents
* Multiple databases without justification
* Enterprise-scale infrastructure

Complexity should only be introduced when justified by a requirement, experiment, reliability concern, security requirement, or deployment need.

---

## 7. Definition of Done

A phase is complete when:

* Implementation is complete.
* Relevant tests pass.
* Required evaluation is performed.
* Documentation is updated.
* Known limitations are recorded.
* Acceptance criteria are satisfied.
* Human review is complete.

---

## 8. Project Principle

> **Build the simplest system that can prove the next technical hypothesis.**

```

This is the version I recommend you **actually put in the repo**. It is enough to control the project without turning `project-plan.md` into a duplicate of your other documents. Your existing problem statement already defines the MVP/V1 scope and metrics, while the architecture and workflows documents already contain the detailed technical design. 
```
