"""Gold queries and corpus definitions for DocuLens AI evaluation benchmark."""

from typing import Any

DOCUMENTS_SPEC: list[dict[str, Any]] = [
    {
        "filename": "doculens-architecture-v1.pdf",
        "pages": [
            {
                "title": "DocuLens AI Enterprise Architecture — Overview & Core Layers",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1.1 System Overview",
                        "content": "DocuLens AI consists of five major architectural layers: (1) Frontend, (2) API/Application Layer, (3) Document Intelligence Layer, (4) Retrieval and Generation Layer, and (5) Infrastructure Layer. The system is architected for strict evidence traceability, deterministic citation validation, and high-performance multimodal retrieval.",
                        "height": 55,
                    },
                    {
                        "heading": "1.2 Document Intelligence Components",
                        "content": "The Document Intelligence Layer uses PyMuPDF for PDF parsing, page geometry analysis, and page image rendering. Text chunking uses semantic structure boundaries with configurable chunk size and overlap.",
                        "height": 45,
                    },
                    {
                        "heading": "1.3 Phase 4 — Hybrid Retrieval Fusion",
                        "content": "The hybrid retrieval step combines Dense Retrieval (BGE embeddings), BM25 (sparse lexical), and Visual Retrieval (CLIP vision embeddings) using either Reciprocal Rank Fusion (RRF) or normalized weighted score fusion.",
                        "height": 50,
                    },
                    {
                        "type": "figure_box",
                        "fig_title": "Figure 1: Multimodal Ingestion and Retrieval Pipeline Diagram",
                        "content": "Ingestion flow: Upload PDF -> PyMuPDF Page Parsing -> Dense Chunker + Visual Page Renderer (150 DPI) -> Qdrant Vector Collections (document_chunks & visual_pages) + In-Memory BM25 Index. Query flow executes parallel dense/sparse/visual search, applies RRF fusion, and executes cross-encoder reranking.",
                        "height": 70,
                    },
                ],
            },
            {
                "title": "DocuLens AI Engineering Standards — Traceability & Quality Gates",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "5. Document Intelligence & Traceability Standards",
                        "content": "For complete evidence traceability across the pipeline, required metadata fields must include where applicable: document_id, page_number, chunk_id, and source/evidence reference. Chunks without valid page associations are rejected during validation.",
                        "height": 55,
                    },
                    {
                        "heading": "6. Quality Gates and Grounded Generation",
                        "content": "Deterministic citation validation ensures strict grounding against retrieved evidence. The AnswerGenerator inspects all evidence tags (e.g. [Evidence 1]) and verifies that referenced page numbers match actual retrieved chunks. Answers with missing evidence or hallucinated citations are flagged as ungrounded.",
                        "height": 60,
                    },
                    {
                        "heading": "7. Rate Limiting and Telemetry",
                        "content": "In-process fixed-window rate limiting is enforced per client IP with a default window of 60.0 seconds and maximum 60 requests. System health metrics track pipeline latency percentiles (p50, p95) and citation verification rates.",
                        "height": 55,
                    },
                ],
            },
        ],
    },
    {
        "filename": "sat-telemetry-spec-b2.pdf",
        "pages": [
            {
                "title": "Satellite Subsystem Telemetry Specification SAT-REF-77",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Telemetry Subsystem Scope",
                        "content": "Subsystem Specification SAT-REF-77-OMEGA governs telemetry transmission rates, downlink frequency bands, and onboard packet buffering for the Leo-Sat orbital platform. The primary telemetry downlink operates on the Ka-band frequency of 14.25 GHz with a maximum bandwidth of 350 MHz.",
                        "height": 55,
                    },
                    {
                        "heading": "2. Transmission Protocols and Error Codes",
                        "content": "Under nominal conditions, packet loss must not exceed 0.001%. If orbital drift exceeds 0.05 degrees, the attitude controller raises error code ERR_ORBIT_9921, initiating an automated thruster realignment burn.",
                        "height": 50,
                    },
                ],
            },
            {
                "title": "Satellite Subsystem Telemetry Specification SAT-REF-77 — Budget & Power",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Fiscal Allocation and Power Consumption",
                        "content": "Total allocated orbital budget for the SAT-REF-77-OMEGA subsystem is $8,500,000 for fiscal cycle 2026. Nominal operating power consumption is 320 Watts during downlink and 45 Watts during solar occultation standby mode.",
                        "height": 55,
                    },
                    {
                        "type": "table",
                        "content": "Subsystem Component | Power (W) | Weight (kg) | Operating Temp\nTransponder Array   | 180 W     | 12.4 kg     | -20C to +55C\nDiplexer Unit       | 40 W      | 4.2 kg      | -40C to +70C\nHigh-Gain Antenna   | 100 W     | 18.5 kg     | -50C to +85C",
                        "height": 75,
                    },
                ],
            },
        ],
    },
    {
        "filename": "financial-quarterly-q3.pdf",
        "pages": [
            {
                "title": "NovaTech Holdings — Q3 Financial Performance & Revenue",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Executive Summary",
                        "content": "NovaTech Holdings reported total revenue of $89.4 million for the third quarter of 2026, representing a 14.5% year-over-year growth. Growth was primarily driven by the Cloud Platform division.",
                        "height": 45,
                    },
                    {
                        "type": "table",
                        "content": "Division       | Q3 Revenue | YoY Growth | Operating Margin\nCloud Platform | $45.2M     | +28.4%     | 34.2%\nEnterprise SW  | $31.8M     | +6.1%      | 28.0%\nHardware/Edge  | $12.4M     | -3.2%      | 14.5%\nTotal          | $89.4M     | +14.5%     | 29.2%",
                        "height": 95,
                    },
                ],
            },
            {
                "title": "NovaTech Holdings — Regional Breakdown & Operating Expenses",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "2. Regional Performance",
                        "content": "North America contributed $52.6 million (58.8% of total revenue), EMEA contributed $24.1 million (27.0%), and Asia-Pacific accounted for $12.7 million (14.2%). Research and Development (R&D) expenses rose to $18.6 million, reflecting increased investments in autonomous AI workflow engines.",
                        "height": 65,
                    },
                    {
                        "heading": "3. EBITDA and Net Income",
                        "content": "Adjusted EBITDA for Q3 reached $26.1 million with an EBITDA margin of 29.2%. Net income was $16.8 million, up from $14.1 million in Q3 of the prior fiscal year.",
                        "height": 50,
                    },
                ],
            },
            {
                "title": "NovaTech Holdings — Risk Disclosures & Capital Expenditure",
                "page_num": 3,
                "blocks": [
                    {
                        "heading": "4. Capital Expenditures and Cash Reserves",
                        "content": "Capital expenditures in Q3 were $7.8 million, dedicated to GPU datacenter expansion in Northern Virginia. Total cash and cash equivalents stood at $142.5 million as of September 30, 2026.",
                        "height": 55,
                    },
                    {
                        "heading": "5. Key Risk Disclosures",
                        "content": "Supply chain constraints in specialized semiconductor packaging remain the primary operational risk. Cloud customer retention rate remained strong at 96.4% across tier-1 enterprise contracts.",
                        "height": 50,
                    },
                ],
            },
        ],
    },
    {
        "filename": "database-engine-manual.pdf",
        "pages": [
            {
                "title": "HyperStore DB Architecture & LSM-Tree Engine Manual",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. LSM-Tree Compaction Architecture",
                        "content": "HyperStore DB utilizes a Log-Structured Merge (LSM) storage engine optimized for high-throughput write workloads. MemTables are flushed to Level-0 SSTables when active buffer size reaches 64 MB. Level-0 allows up to 4 overlapping files before triggering Level-0 to Level-1 compaction.",
                        "height": 65,
                    },
                    {
                        "heading": "2. Compaction Size Multipliers",
                        "content": "Each level from Level-1 upward uses a 10x size amplification multiplier. Level-1 capacity is capped at 100 MB, Level-2 at 1,000 MB (1 GB), and Level-3 at 10 GB.",
                        "height": 50,
                    },
                ],
            },
            {
                "title": "HyperStore DB — WAL and Read Path Optimization",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Write-Ahead Log (WAL) Durability",
                        "content": "All write transactions append to a synchronous Write-Ahead Log. Under the strict fsync policy, WAL write latency SLA is strictly guaranteed under 5.0 milliseconds. Group commit batching coalesces up to 32 concurrent writes per fsync cycle.",
                        "height": 60,
                    },
                    {
                        "heading": "4. Bloom Filter Indexing",
                        "content": "SSTable blocks include Block-level Bloom filters configured with 10 bits per key, achieving a target false positive rate of 1.0%. This prevents unnecessary disk seeks on point lookups for missing keys.",
                        "height": 55,
                    },
                ],
            },
        ],
    },
    {
        "filename": "clinical-trial-protocol.pdf",
        "pages": [
            {
                "title": "MedVance Protocol MV-804 — Phase II Clinical Trial",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Study Design and Objectives",
                        "content": "Protocol MV-804 is a randomized, double-blind, placebo-controlled Phase II trial evaluating the efficacy and safety of drug candidate MV-804 in adult patients diagnosed with Type 2 Diabetes Mellitus with inadequate glycemic control.",
                        "height": 55,
                    },
                    {
                        "heading": "2. Patient Inclusion and Exclusion Criteria",
                        "content": "Inclusion criteria: Age between 18 and 65 years inclusive; baseline HbA1c between 7.5% and 10.0%; Body Mass Index (BMI) between 24.0 and 38.0 kg/m2. Exclusion criteria: History of severe cardiovascular events within 6 months prior to screening, or estimated GFR < 45 mL/min/1.73m2.",
                        "height": 65,
                    },
                ],
            },
            {
                "title": "MedVance Protocol MV-804 — Dosing and Endpoints",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Dosing Regimen and Administration",
                        "content": "Subjects are randomized 1:1:1 to receive oral MV-804 25 mg twice daily (BID), MV-804 50 mg twice daily (BID), or matching placebo for a total treatment duration of 24 weeks.",
                        "height": 55,
                    },
                    {
                        "heading": "4. Primary and Secondary Endpoints",
                        "content": "The primary efficacy endpoint is the mean change in HbA1c from baseline to Week 24. Secondary endpoints include the proportion of patients achieving HbA1c < 7.0%, changes in fasting plasma glucose (FPG), and body weight changes at Week 12 and Week 24.",
                        "height": 65,
                    },
                ],
            },
        ],
    },
    {
        "filename": "cloud-security-compliance.pdf",
        "pages": [
            {
                "title": "ApexCloud SOC 2 Type II Security Standard & IAM Policy",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Identity and Access Management (IAM) Policy",
                        "content": "ApexCloud enforces mandatory Multi-Factor Authentication (MFA) across all administrative accounts. Passwords must be a minimum of 16 characters in length, include uppercase, lowercase, numbers, and special symbols, and must be rotated every 90 days. Inactive sessions time out after 15 minutes.",
                        "height": 65,
                    },
                    {
                        "heading": "2. Data Encryption Standards",
                        "content": "All data at rest is encrypted using AES-256-GCM encryption with envelope keys managed by AWS KMS or HashiCorp Vault. Data in transit requires TLS 1.3 encryption with strict cipher suite enforcement (no CBC or RC4 suites allowed).",
                        "height": 60,
                    },
                ],
            },
            {
                "title": "ApexCloud Security Standard — Audit & Incident Response",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Audit Logging and Retention",
                        "content": "All API calls, authentication attempts, and privilege escalations are logged immutably to WORM-compliant storage. Security audit logs must be retained for a mandatory minimum period of 7 years in compliance with SOC 2 Type II and FedRAMP standards.",
                        "height": 60,
                    },
                    {
                        "heading": "4. Incident Classification SLA",
                        "content": "P1 Critical security incidents require an initial containment response within 15 minutes of detection and executive briefing within 1 hour.",
                        "height": 45,
                    },
                ],
            },
        ],
    },
    {
        "filename": "network-switch-specs.pdf",
        "pages": [
            {
                "title": "NetCore Nexus-9000 Data Center Switch Datasheet",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Hardware Specifications & Port Density",
                        "content": "The NetCore Nexus-9000 is a high-density, low-latency 1RU data center switch featuring 48x 100GbE QSFP28 ports and 8x 400GbE QSFP-DD uplink ports. It delivers a total non-blocking switching capacity of 6.4 Tbps and a packet forwarding rate of 4.7 Bpps.",
                        "height": 65,
                    },
                    {
                        "type": "table",
                        "content": "Parameter          | Specification\nSwitching Capacity | 6.4 Tbps (Full Duplex)\nForwarding Rate    | 4.7 Billion Packets/sec\nPort Configuration | 48x 100G QSFP28 + 8x 400G QSFP-DD\nBuffer Size        | 64 MB Shared Dynamic Buffer\nLatency            | 450 nanoseconds (cut-through)",
                        "height": 95,
                    },
                ],
            },
            {
                "title": "NetCore Nexus-9000 — Power, Cooling & Reliability",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "2. Power and Cooling Requirements",
                        "content": "Maximum power consumption under full 100% traffic load is 450 Watts, with a typical idle power draw of 185 Watts. The chassis supports 1+1 redundant hot-swappable 80-PLUS Platinum power supplies and 5+1 redundant fan trays with front-to-back airflow.",
                        "height": 65,
                    },
                    {
                        "heading": "3. Reliability Metrics",
                        "content": "Mean Time Between Failures (MTBF) is rated at 350,000 operating hours at 25C ambient temperature. The system operates within an ambient temperature envelope of 0C to 40C.",
                        "height": 50,
                    },
                ],
            },
        ],
    },
    {
        "filename": "solar-energy-research.pdf",
        "pages": [
            {
                "title": "Heliostat Thermal Efficiency and Photovoltaic Output Study",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Abstract and Environmental Parameters",
                        "content": "This study analyzes the combined thermal and electrical conversion efficiency of dual-axis tracking heliostat arrays under the standard solar irradiance model AM1.5D (Direct solar irradiance 1000 W/m2 at 25C).",
                        "height": 55,
                    },
                    {
                        "heading": "2. Thermal Receiver Efficiency",
                        "content": "The molten salt central receiver attained a peak thermal conversion efficiency of 41.2% at a working temperature of 650C. Receiver heat loss was mitigated using high-emissivity ceramic coatings.",
                        "height": 55,
                    },
                ],
            },
            {
                "title": "Heliostat Study — Thermal Storage and Capacity Factor",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Molten Salt Thermal Energy Storage (TES)",
                        "content": "The two-tank indirect molten salt thermal storage system provides up to 8.0 hours of full-load turbine generation buffer, enabling high capacity factor operation during non-daylight hours.",
                        "height": 55,
                    },
                    {
                        "type": "figure_box",
                        "fig_title": "Figure 2: Daily Power Output Profile with Thermal Storage",
                        "content": "The chart illustrates baseline solar generation peaking at 13:00 (120 MWth) followed by molten salt dispatch sustaining 85 MWe electrical output from 17:00 to 01:00. Overall plant capacity factor increased from 27.5% to 58.4%.",
                        "height": 75,
                    },
                ],
            },
        ],
    },
    {
        "filename": "distributed-consensus-raft.pdf",
        "pages": [
            {
                "title": "Distributed Consensus — Raft Protocol Implementation Notes",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Leader Election Timing",
                        "content": "In the Raft consensus protocol, follower election timeouts are randomized between 150 ms and 300 ms to prevent split-vote scenarios. If a follower receives no heartbeat AppendEntries RPC within this window, it transitions to Candidate state and increments the current term.",
                        "height": 65,
                    },
                    {
                        "heading": "2. Heartbeat Frequency",
                        "content": "The cluster leader broadcasts empty AppendEntries RPC heartbeats every 50 ms to maintain leadership authority and prevent followers from timing out.",
                        "height": 45,
                    },
                ],
            },
            {
                "title": "Distributed Consensus — Quorum & Log Replication",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Quorum Requirements and Split-Brain Mitigation",
                        "content": "For a cluster of N nodes, consensus requires a strict majority quorum formula of (N/2) + 1 nodes. In a 5-node cluster, at least 3 nodes must acknowledge a log entry before it is committed to state machine.",
                        "height": 55,
                    },
                    {
                        "heading": "4. Log Compaction and Snapshots",
                        "content": "When log entries exceed 10,000 records, the Raft engine triggers asynchronous state machine snapshotting, compacting preceding log entries and discarding applied index history.",
                        "height": 50,
                    },
                ],
            },
        ],
    },
    {
        "filename": "microservices-incident-postmortem.pdf",
        "pages": [
            {
                "title": "Incident Post-Mortem INC-4409 — Authentication Service Failure",
                "page_num": 1,
                "blocks": [
                    {
                        "heading": "1. Incident Summary & Impact",
                        "content": "On August 14, 2026, Incident INC-4409 resulted in a 42-minute partial service degradation across the customer checkout gateway. Approximately 14,200 payment verification requests were dropped with HTTP 504 gateway timeouts.",
                        "height": 60,
                    },
                    {
                        "heading": "2. Root Cause Analysis",
                        "content": "The root cause was connection pool exhaustion in AuthService due to an unindexed database query on the user_sessions table during peak token refresh bursts. Maximum pool limit of 50 connections was saturated within 3 minutes.",
                        "height": 60,
                    },
                ],
            },
            {
                "title": "Incident Post-Mortem INC-4409 — Remediation Actions",
                "page_num": 2,
                "blocks": [
                    {
                        "heading": "3. Corrective Actions and Preventative Measures",
                        "content": "Remediation items completed: (1) Added composite index on user_sessions(user_id, expires_at), reducing query latency from 850ms to 4ms; (2) Increased connection pool cap to 200 with dynamic backpressure; (3) Deployed Netflix Resilience4j circuit breaker pattern to fail fast under downstream degradation.",
                        "height": 75,
                    },
                    {
                        "heading": "4. Incident Timeline",
                        "content": "14:02 UTC - First alert triggered; 14:12 UTC - Incident Commander engaged; 14:35 UTC - Hotfix index applied; 14:44 UTC - Full traffic recovery confirmed.",
                        "height": 50,
                    },
                ],
            },
        ],
    },
]
