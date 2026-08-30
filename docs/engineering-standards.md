# DocuLens AI — Engineering Standards

## 1. Core Principle

Follow:

Understand
→ Design
→ Implement
→ Test
→ Measure
→ Validate
→ Document
→ Improve

Prefer the simplest solution that satisfies the current requirement.

Do not introduce technology, abstraction, infrastructure, or agents without a clear reason.

---

## 2. Code Quality

- Write clear, readable Python and TypeScript.
- Keep functions and modules focused.
- Prefer explicit logic over unnecessary abstractions.
- Use type hints in Python.
- Validate external inputs.
- Keep configuration separate from application logic.
- Avoid duplicated business logic.
- Remove unused code and dependencies.
- Do not optimize prematurely.

---

## 3. Architecture

- Respect the boundaries defined in `docs/architecture.md`.
- Keep deterministic document processing and retrieval logic explicit.
- Keep AI provider calls behind a provider interface/adapter where needed.
- Avoid unnecessary microservices.
- Avoid introducing additional databases without a demonstrated requirement.
- Significant architecture changes require documentation and review.

---

## 4. AI/ML Standards

Every major AI component must have:

- A clear purpose.
- A reason for being used.
- Defined inputs and outputs.
- An evaluation strategy.
- Known limitations.
- Failure handling where appropriate.

Build a baseline before adding advanced retrieval or multimodal capabilities.

Never claim an improvement without measured evidence.

---

## 5. Document Intelligence

Document processing must preserve traceability.

Important metadata should include where applicable:

- `document_id`
- `page_number`
- `chunk_id`
- source/evidence reference

Retrieved evidence must be traceable back to the original document.

---

## 6. Testing

Use testing appropriate to the current phase.

### Unit Tests

Test:

- parsing logic
- chunking
- validation
- retrieval utilities
- schemas
- business logic

### Integration Tests

Test interactions such as:

- API → storage
- ingestion → indexing
- query → retrieval
- retrieval → generation

### AI Evaluation

Software tests do not prove AI quality.

Use evaluation datasets and metrics for:

- retrieval quality
- answer quality
- grounding
- failure cases

---

## 7. Error Handling

Failures must be explicit and controlled.

Handle relevant failures including:

- invalid files
- corrupted PDFs
- extraction failures
- rendering failures
- empty retrieval
- model/API failures
- timeouts
- malformed responses
- insufficient evidence

Do not silently hide important failures.

---

## 8. Configuration and Secrets

- Use environment variables for secrets and environment-specific configuration.
- Maintain `.env.example`.
- Never commit real API keys.
- Never log raw credentials.
- Never expose server-side provider credentials unnecessarily to the frontend.
- Keep provider credentials isolated between users when multi-user functionality exists.

---

## 9. Logging and Observability

Use structured logs where practical.

Track relevant information such as:

- request ID
- document ID
- query ID
- processing time
- retrieval latency
- generation latency
- model/provider
- token usage where available
- failure type

Do not log sensitive document content or credentials unnecessarily.

---

## 10. Git

Use feature branches for significant changes.

```text
main
 └── feature/<change>
````

Before merging:

* review the diff
* run relevant tests
* verify documentation
* check for accidental changes
* confirm secrets are not committed

Do not casually make significant changes directly on `main`.

---

## 11. AI Coding Agents

AI coding agents are implementation assistants, not project owners.

Agents may help with:

* implementation
* boilerplate
* tests
* refactoring
* documentation
* debugging

The project owner remains responsible for:

* architecture
* technology decisions
* evaluation
* security
* validation
* final review
* portfolio claims

Generated code must be reviewed before being considered correct.

---

## 12. Security

Apply security according to the current capabilities of the system.

Relevant controls include:

* file validation
* file-size limits
* input validation
* secret protection
* authorization
* document isolation
* prompt-injection protection
* restricted tool access
* safe error handling
* sensitive-data exclusion from logs

Do not add unnecessary enterprise security infrastructure before it is required.

---

## 13. Agentic Systems

If product agents are introduced:

* Use explicit tools.
* Define tool schemas.
* Restrict permissions.
* Limit retries/iterations.
* Validate tool outputs.
* Prevent unrestricted filesystem/database/shell access.
* Log important agent decisions.
* Prefer deterministic workflows when they are more reliable.

Agents must solve a demonstrated problem rather than exist only as a technology showcase.

---

## 14. Documentation

Documentation should explain real engineering decisions.

Maintain documentation when needed for:

* architecture
* workflows
* API contracts
* evaluation
* testing
* security
* deployment
* important architecture decisions
* experiments

Do not create documentation merely to make the repository appear larger.

---

## 15. Definition of Done

A change is complete when:

* Implementation is complete.
* Relevant tests pass.
* Required evaluation is performed.
* Documentation is updated.
* Known limitations are understood.
* No unnecessary scope was introduced.
* The change has been reviewed.

```text
Code
 ↓
Test
 ↓
Measure
 ↓
Review
 ↓
Document
 ↓
Done
```

---

## 16. Engineering Rule

> Every technology, model, abstraction, and architectural component must have a defensible reason for existing.

```

**This is enough for now.** Don't add more sections to this file yet. The detailed testing, evaluation, security, API, and deployment documents should be created **when those parts become relevant**, rather than duplicating them here. Your source guidance explicitly recommends proportional documentation and warns against creating documentation merely to make the repository look large. :contentReference[oaicite:1]{index=1}
```
