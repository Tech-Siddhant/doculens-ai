````markdown
# DocuLens AI — Product & UI Design Specification

> Single source of truth for the DocuLens AI frontend design.

## 1. Product Identity

### Product
**DocuLens AI**

### One-line description
A multimodal document intelligence platform that retrieves evidence from complex PDFs and produces grounded answers with traceable page citations.

### Core product idea

Upload Document
→ Understand Document
→ Retrieve Evidence
→ Ground Answer
→ Show Sources

The product must NOT feel like a generic "Chat with PDF" application.

Its identity comes from:

Document Intelligence
+
Multimodal Retrieval
+
Grounded Answers
+
Page-Level Citations
+
Optional Technical Transparency

---

## 2. Design Philosophy

### Principle

**Simple by default. Transparent on demand.**

The normal user should see only what is necessary:

- document
- question
- answer
- citations
- relevant page/evidence

Technical system details remain hidden unless the user explicitly enables them.

The frontend is a presentation/client layer.

It must NOT:

- perform vector search
- access Qdrant directly
- access BM25 directly
- execute retrieval algorithms
- contain LLM orchestration
- contain provider API secrets
- expose API keys
- reproduce backend business logic

All AI/retrieval behavior belongs to the FastAPI backend.

---

## 3. Target Users

Primary users:

- Researchers
- Engineers
- Students
- Technical analysts
- Professionals working with technical documents

Initial product domain:

**Research and technical PDFs**

---

## 4. Visual Direction

Create a clean, minimal, premium technical aesthetic.

Visual references:

- Linear
- Vercel
- modern developer tools
- modern research applications

Do not copy any existing product.

Use them only as visual-quality references.

The interface should feel:

- minimal
- modern
- technical
- calm
- premium
- trustworthy
- highly readable
- professional

---

## 5. Visual Rules

### Color

Use a restrained neutral palette.

Background:
- near-white / very light neutral

Surfaces:
- white or slightly elevated neutral surfaces

Primary text:
- dark charcoal rather than pure black

Secondary text:
- muted gray

Borders:
- subtle gray

Accent:
- one restrained accent color used consistently

Status colors:

Green:
- success
- valid
- ready

Amber:
- warning
- fallback
- degraded

Red:
- error
- failed
- rejected

Do not use status colors decoratively.

---

## 6. Typography

Use a clean modern sans-serif typeface.

Hierarchy:

Page title
→ Section heading
→ Question
→ Answer
→ Evidence
→ Metadata

Prioritize readability over visual decoration.

Avoid:

- oversized headings
- excessive font weights
- excessive uppercase text
- decorative typography

---

## 7. Layout Philosophy

Prefer whitespace over excessive cards.

Avoid turning every element into a card.

Use:

- spacing
- subtle dividers
- restrained borders
- clear hierarchy

The application should feel spacious without wasting screen area.

---

## 8. Main Application Structure

Desktop:

```text
┌─────────────────────────────────────────────────────────────┐
│ Header                                                      │
├──────────────┬──────────────────────────────┬───────────────┤
│              │                              │               │
│ Sidebar      │ Main Workspace               │ Document      │
│              │                              │ Viewer        │
│ Documents    │ Question                     │               │
│ Recent       │                              │ Page          │
│ Settings     │ Answer                       │ Preview       │
│              │                              │               │
│              │ Citations                    │               │
│              │                              │               │
└──────────────┴──────────────────────────────┴───────────────┘
````

On smaller screens:

```text
Header
↓
Document
↓
Question
↓
Answer
↓
Citations
↓
Document / Evidence
```

The technical pipeline becomes a drawer or bottom sheet.

---

## 9. Navigation

### Sidebar

Primary navigation:

* Dashboard
* Documents
* New Document
* Settings

Optional:

* Recent documents

Keep the sidebar visually lightweight.

Do not create a complex enterprise navigation system.

---

## 10. Dashboard

The dashboard should immediately communicate:

**"What can I do here?"**

Primary actions:

* Upload a document
* Open a recent document

Display:

* recent documents
* processing status
* quick access to documents

Avoid:

* fake analytics
* meaningless statistics
* decorative charts
* excessive KPI cards

---

## 11. Document Library

Create a clean document list/table.

Each document displays:

* document name
* page count
* status
* upload date
* actions

Example:

```text
┌─────────────────────────────────────────────────────────────┐
│ Research Paper.pdf                    Ready      12 pages  │
│ Uploaded today                                      •••     │
├─────────────────────────────────────────────────────────────┤
│ Technical Report.pdf                  Processing   8 pages │
│ Uploaded yesterday                                  •••     │
└─────────────────────────────────────────────────────────────┘
```

Include:

* search
* upload action
* empty state
* processing state
* failure state

---

## 12. Upload Experience

The upload experience should be extremely simple.

Primary drop zone:

```text
Drop your PDF here

or

Choose file
```

Supporting text:

```text
PDF files up to the configured upload limit
```

After upload:

```text
Uploading
↓
Validating
↓
Processing
↓
Indexing
↓
Ready
```

Use actual backend status when available.

Never invent progress percentages.

---

## 13. Processing State

Use a progressive status display.

Example:

```text
Processing document

✓ Document uploaded
✓ Document validated
✓ Text extracted
● Building searchable index
○ Finalizing document
```

Communicate progress without pretending to know exact completion percentages.

---

## 14. Document Viewer

The document viewer is one of the primary product surfaces.

It should support:

* page thumbnails
* page navigation
* current page
* zoom
* citation navigation
* highlighted source page where supported

Layout:

```text
┌────────────┬──────────────────────────────┐
│ Thumbnails │                              │
│            │                              │
│ Page 1     │          PDF Page            │
│ Page 2     │                              │
│ Page 3     │                              │
│ Page 4     │                              │
│            │                              │
└────────────┴──────────────────────────────┘
```

When a citation is clicked:

Citation
↓
Relevant page
↓
Page becomes active
↓
Relevant evidence is emphasized when available

---

## 15. Question & Answer Interface

Primary input:

```text
Ask anything about this document...
```

Example questions:

```text
What methodology did the authors use?

Which model performed best according to Table 2?

What does Figure 3 demonstrate?

What limitations did the authors identify?
```

The question interface should remain focused.

Avoid:

* unnecessary chat bubbles
* avatars
* decorative AI animations
* fake "thinking" animations

---

## 16. Answer Presentation

The answer should be the primary content.

Structure:

```text
ANSWER

Concise grounded response...

SOURCES

[Page 4] [Page 7] [Page 9]
```

Clearly distinguish:

1. Answer
2. Sources
3. Evidence

Do not overwhelm the user with retrieval metadata by default.

---

## 17. Citation Design

Citations are a core product feature.

Use compact citation chips:

```text
[Page 4]
[Page 7]
[Page 9]
```

or:

```text
Evidence 1 · Page 4
Evidence 2 · Page 7
```

Citation interaction:

```text
Click citation
↓
Open document viewer
↓
Navigate to cited page
↓
Highlight evidence when available
```

Citation metadata must be traceable to backend evidence.

The frontend must never fabricate citations.

---

## 18. Visual Evidence

When the answer depends on visual evidence, display:

```text
Visual Evidence
────────────────────────

Page 7

[Page image preview]

Source: Visual retrieval
Status: Valid
```

Clearly distinguish visual evidence from textual evidence.

---

## 19. Signature Feature — Show Pipeline

The most important differentiating interaction is:

**Show pipeline**

Default:

```text
OFF
```

When OFF:
show a clean user-facing product experience.

When ON:
reveal observable system behavior.

---

## 20. Security / UX Rule

Do NOT call this:

**Show reasoning**

Prefer:

**Show pipeline**

or:

**Show system details**

The UI must never expose:

* hidden chain-of-thought
* private model reasoning
* internal prompts
* secrets
* API keys

It may expose observable system telemetry.

---

## 21. Technical Pipeline View

When enabled:

```text
Query
↓
Hybrid Retrieval
↓
Reranking
↓
Evidence Selection
↓
Evidence Validation
↓
Context Assembly
↓
LLM Generation
↓
Citation Validation
```

Represent this as a compact vertical timeline.

---

## 22. Pipeline Stage Component

Each stage displays:

* stage name
* latency
* short observable description
* status
* input count
* output count

Example:

```text
RETRIEVAL                              42 ms

Dense + BM25 + Visual
24 candidates

● Success
```

Another:

```text
RERANKING                              18 ms

Cross-encoder
Top 8 candidates

● Success
```

Another:

```text
EVIDENCE VALIDATION                     3 ms

8 checked · 7 valid · 1 rejected

● Warning
```

---

## 23. Pipeline Status

Use:

Green:

* Success

Amber:

* Fallback / Warning

Red:

* Failed

Gray:

* Skipped

Do not make legitimate fallbacks look like catastrophic failures.

---

## 24. Pipeline Metrics

Show only useful observable metrics.

Potential metrics:

* Total latency
* Retrieval latency
* Number of candidates
* Selected evidence count
* Reranking status
* Context size
* Generation latency
* Citation count
* Citation validation status

Do not display metrics that the backend does not actually provide.

---

## 25. Evidence Inspector

When technical mode is enabled, allow inspection of retrieved evidence.

Example:

```text
EVIDENCE 01

Page 7
Score 0.87
Source: Dense + BM25
Status: Valid

Short evidence preview...
```

Another:

```text
EVIDENCE 02

Page 9
Source: Visual
Status: Valid

[Page image]
```

The inspector should help engineers understand:

* What was retrieved?
* Why was it selected?
* Was it validated?
* Where did it come from?

It must NOT pretend to expose hidden model reasoning.

---

## 26. Technical Mode UX

Default:

```text
Question
↓
Answer
↓
Sources
```

Technical mode:

```text
Question
↓
Answer
↓
Sources
↓
Show Pipeline
↓
Retrieval
↓
Evidence Inspector
↓
Validation
```

The technical layer should remain optional.

---

## 27. Loading States

Document processing:

```text
Uploading...
Validating...
Extracting...
Indexing...
Ready
```

Question answering:

```text
Searching document...
Retrieving evidence...
Reranking evidence...
Validating sources...
Generating grounded answer...
```

Do not show fake model thoughts.

---

## 28. Empty States

### No documents

```text
No documents yet

Upload a technical PDF to get started.

[Upload document]
```

### No question

```text
Ask a question about this document.
```

### No evidence

```text
No reliable evidence was found.

Try asking about a topic covered in this document.
```

---

## 29. Error States

Support clear states for:

* invalid PDF
* upload failure
* processing failure
* retrieval failure
* no relevant evidence
* LLM provider unavailable
* rate limit
* timeout
* citation validation failure

Example:

```text
Unable to generate a grounded answer.

The AI provider is temporarily unavailable.

Please try again.
```

Do not expose:

* stack traces
* API keys
* internal paths
* sensitive backend information

---

## 30. Settings

Settings should remain minimal.

Display:

```text
AI Provider

Provider: Gemini
Model: configured model
Status: Connected
```

Never display actual API keys.

Use:

```text
API key: ••••••••••••
```

or:

```text
API key configured
```

The browser must never receive provider secrets unnecessarily.

---

## 31. Responsive Design

### Desktop

Three-zone experience:

```text
Sidebar | Q&A | Document Viewer
```

### Tablet

Two-zone experience:

```text
Navigation
+
Main workspace
```

### Mobile

Single-column:

```text
Document
Question
Answer
Sources
```

Technical pipeline:

```text
Bottom sheet / drawer
```

---

## 32. Accessibility

Support:

* keyboard navigation
* visible focus states
* readable contrast
* semantic buttons
* meaningful labels
* accessible dialogs/drawers
* screen-reader-friendly status messages
* reduced-motion preference

Do not communicate important state using color alone.

---

## 33. Component System

Reusable components:

```text
AppShell
Sidebar
DocumentList
DocumentRow
UploadZone
ProcessingStatus
DocumentViewer
PageThumbnail
QuestionInput
AnswerView
CitationChip
EvidenceCard
EvidenceInspector
PipelineToggle
PipelineTimeline
PipelineStage
StatusBadge
LoadingState
EmptyState
ErrorState
SettingsPanel
```

Components should remain small and composable.

Avoid abstractions without a real reuse requirement.

---

## 34. Frontend Architecture Boundary

```text
                  Browser
                     │
                     ▼
              Next.js / React
                     │
                     ▼
                API Client
                     │
                     ▼
                  FastAPI
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Document Services       RAG Pipeline
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 Retrieval   Validation  Generation
```

The frontend never directly accesses:

* Qdrant
* BM25
* embedding models
* LLM provider APIs
* filesystem
* secrets

---

## 35. Frontend State Principles

Prefer simple state management.

Use local component state where possible.

Use shared state only when multiple views genuinely need the same state.

Do not introduce a global state framework merely because it is popular.

Important states include:

```text
selectedDocument
uploadStatus
processingStatus
question
answer
citations
selectedPage
pipelineVisible
pipelineTrace
error
```

---

## 36. API Integration Principle

The frontend consumes backend contracts.

Do not duplicate backend schemas manually when generated or shared contracts
are available.

The API client should centralize:

* base URL
* request handling
* error handling
* response parsing
* authentication mechanism when introduced later

---

## 37. Security Principles

Never place secrets in:

* React components
* browser local storage
* public environment variables
* client-side configuration
* source code

Never expose:

* provider API keys
* filesystem paths
* internal database credentials
* internal prompts
* stack traces

The frontend is not a security boundary.

---

## 38. Performance Principles

Prioritize:

* fast initial render
* lazy loading of heavy document viewer components
* efficient page rendering
* avoiding unnecessary re-renders
* limited client-side dependencies
* no unnecessary polling
* no unnecessary large assets

Do not sacrifice functionality merely for micro-optimizations.

Measure before optimizing.

---

## 39. Animation

Use animation sparingly.

Appropriate:

* panel opening
* drawer transitions
* citation navigation
* subtle loading indicators
* status transitions

Avoid:

* continuous decorative animations
* flashy particle effects
* excessive motion
* fake AI "thinking" animations

Respect reduced-motion preferences.

---

## 40. Design Tokens

Maintain centralized tokens for:

* colors
* spacing
* typography
* radius
* shadows
* borders
* transitions

Example spacing scale:

```text
4
8
12
16
24
32
48
64
```

Keep the scale small and consistent.

---

## 41. Product Personality

DocuLens should feel:

```text
Precise
Calm
Technical
Trustworthy
Transparent
Useful
```

It should NOT feel:

```text
Flashy
Toy-like
Over-engineered
Generic
AI-hype driven
```

---

## 42. Demo Experience

A recruiter should understand the product within approximately one minute.

Ideal demo:

```text
1. Upload PDF
       ↓
2. Open document
       ↓
3. Ask question
       ↓
4. Receive grounded answer
       ↓
5. Click citation
       ↓
6. Page opens
       ↓
7. Toggle "Show pipeline"
       ↓
8. Inspect retrieval → evidence → validation
```

The demo should demonstrate both:

### Product value

"Can it answer my document question?"

and:

### Engineering depth

"Can I see how the system retrieved and validated the evidence?"

---

## 43. What the UI Must Demonstrate

The UI should make these capabilities visible:

```text
PDF ingestion
↓
Document understanding
↓
Multimodal retrieval
↓
Hybrid evidence
↓
Reranking
↓
Evidence validation
↓
Grounded generation
↓
Citation validation
```

Only the appropriate amount of technical detail should be visible
to the user.

---

## 44. What the UI Must NOT Claim

Do not display:

* "100% accurate"
* "hallucination-free"
* "production-ready"
* fabricated latency
* fabricated retrieval scores
* fabricated confidence
* fabricated model reasoning
* fabricated system status

All metrics must originate from actual backend measurements.

---

## 45. Design Acceptance Criteria

The frontend design is successful when:

### Visual

* clean
* minimal
* consistent
* readable
* responsive
* accessible

### Product

A user can:

```text
Upload PDF
→ Find document
→ Open document
→ Ask question
→ Read answer
→ Click citation
→ Inspect page
```

### Technical transparency

A technical user can:

```text
Enable pipeline
→ inspect stages
→ inspect observable metrics
→ inspect evidence
→ inspect citation validation
```

### Security

The frontend never exposes:

* API keys
* secrets
* internal filesystem paths
* internal prompts
* database credentials

### Architecture

The frontend remains a client of the FastAPI API and does not implement
backend AI/retrieval logic.

---

## 46. Core UX Principle

The most important UX decision in DocuLens is:

> **Hide complexity until the user asks to see it.**

Normal user:

```text
Question
↓
Answer
↓
Sources
```

Technical user:

```text
Question
↓
Answer
↓
Sources
↓
Show Pipeline
↓
Retrieval
↓
Reranking
↓
Evidence
↓
Validation
↓
Generation
↓
Citation Validation
```

This is the core visual identity of DocuLens AI.

```
```
