"""Compiles and writes gold benchmark dataset queries."""

import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "gold_dataset.jsonl"


def create_query(
    qid: str,
    doc_id: str,
    question: str,
    category: str,
    pages: list[int],
    evidence: str,
    answer: str,
    facts: list[str],
    is_answerable: bool = True,
    sources: list[str] | None = None,
    difficulty: str = "easy",
    note: str = "",
) -> dict[str, Any]:
    return {
        "query_id": qid,
        "document_id": doc_id,
        "question": question,
        "category": category,
        "expected_sources": sources or ["text"],
        "ground_truth_pages": pages,
        "ground_truth_chunks": [],
        "ground_truth_evidence_text": evidence,
        "ground_truth_answer": answer,
        "key_reference_facts": facts,
        "is_answerable": is_answerable,
        "difficulty": difficulty,
        "metadata": {"source_file": f"{doc_id}.pdf", "note": note},
    }


def get_queries_part1() -> list[dict[str, Any]]:
    return [
        create_query("q-001", "doculens-architecture-v1", "What are the five major architectural layers of DocuLens AI?", "factoid_text", [1], "DocuLens AI consists of five major architectural layers: (1) Frontend, (2) API/Application Layer, (3) Document Intelligence Layer, (4) Retrieval and Generation Layer, and (5) Infrastructure Layer.", "The five major architectural layers of DocuLens AI are: Frontend, API/Application Layer, Document Intelligence Layer, Retrieval and Generation Layer, and Infrastructure Layer.", ["Frontend", "API/Application Layer", "Document Intelligence Layer", "Retrieval and Generation Layer", "Infrastructure Layer"]),
        create_query("q-002", "doculens-architecture-v1", "Which retrieval channels are combined in the hybrid retrieval step?", "factoid_text", [1], "combines Dense Retrieval (BGE embeddings), BM25 (sparse lexical), and Visual Retrieval (CLIP vision embeddings)", "The hybrid retrieval step combines Dense Retrieval, BM25 (sparse lexical), and Visual Retrieval.", ["Dense Retrieval", "BM25", "Visual Retrieval"]),
        create_query("q-003", "doculens-architecture-v1", "What metadata fields are required for retrieved evidence traceability according to the engineering standards?", "multi_page_reasoning", [2], "required metadata fields must include where applicable: document_id, page_number, chunk_id, and source/evidence reference", "The required metadata fields are document_id, page_number, chunk_id, and source/evidence reference.", ["document_id", "page_number", "chunk_id", "source/evidence reference"], difficulty="medium"),
        create_query("q-004", "doculens-architecture-v1", "What is the training loss curve for the DocuLens embedding model across 100 epochs?", "negative_unanswerable", [], "", "This information is not present in the documentation. DocuLens AI does not train custom embedding models; it uses pre-trained models.", [], is_answerable=False),
        create_query("q-005", "doculens-architecture-v1", "What library is utilized in the Document Intelligence Layer for PDF parsing and page rendering?", "factoid_text", [1], "The Document Intelligence Layer uses PyMuPDF for PDF parsing, page geometry analysis, and page image rendering.", "PyMuPDF is used for PDF parsing, page geometry analysis, and page rendering.", ["PyMuPDF"]),
        create_query("q-006", "doculens-architecture-v1", "Explain the full ingestion and query flow illustrated in Figure 1 of the architecture document.", "figure_chart_analysis", [1], "Figure 1: Multimodal Ingestion and Retrieval Pipeline Diagram: Ingestion flow: Upload PDF -> PyMuPDF Page Parsing -> Dense Chunker + Visual Page Renderer (150 DPI) -> Qdrant Vector Collections (document_chunks & visual_pages) + In-Memory BM25 Index. Query flow executes parallel dense/sparse/visual search, applies RRF fusion, and executes cross-encoder reranking.", "Figure 1 illustrates an ingestion pipeline where uploaded PDFs are parsed with PyMuPDF, chunked for dense embeddings, rendered at 150 DPI for visual page vectors in Qdrant, and indexed in BM25. The query flow executes parallel multi-channel retrieval, applies RRF fusion, and passes results to a cross-encoder reranker.", ["Figure 1", "PyMuPDF", "150 DPI", "Qdrant", "RRF fusion", "cross-encoder"], sources=["text", "visual", "figure"], difficulty="medium"),
        create_query("q-007", "doculens-architecture-v1", "How does DocuLens AI handle rate limiting and what are the default parameters?", "factoid_text", [2], "In-process fixed-window rate limiting is enforced per client IP with a default window of 60.0 seconds and maximum 60 requests.", "DocuLens AI uses in-process fixed-window rate limiting per client IP with a 60.0-second window and a limit of 60 requests.", ["fixed-window", "per client IP", "60.0 seconds", "60 requests"]),
        create_query("q-008", "sat-telemetry-spec-b2", "What frequency band and specific frequency does the SAT-REF-77-OMEGA primary telemetry downlink operate on?", "factoid_text", [1], "The primary telemetry downlink operates on the Ka-band frequency of 14.25 GHz with a maximum bandwidth of 350 MHz.", "The primary telemetry downlink operates on the Ka-band frequency of 14.25 GHz.", ["Ka-band", "14.25 GHz"]),
        create_query("q-009", "sat-telemetry-spec-b2", "Which error code is raised when orbital drift exceeds 0.05 degrees, and what action does it trigger?", "factoid_text", [1], "If orbital drift exceeds 0.05 degrees, the attitude controller raises error code ERR_ORBIT_9921, initiating an automated thruster realignment burn.", "Error code ERR_ORBIT_9921 is raised, which initiates an automated thruster realignment burn.", ["ERR_ORBIT_9921", "automated thruster realignment burn"]),
    ]


def get_queries_part2() -> list[dict[str, Any]]:
    return [
        create_query("q-010", "sat-telemetry-spec-b2", "What is the total allocated orbital budget for the SAT-REF-77-OMEGA subsystem for fiscal cycle 2026?", "factoid_text", [2], "Total allocated orbital budget for the SAT-REF-77-OMEGA subsystem is $8,500,000 for fiscal cycle 2026.", "The total allocated orbital budget is $8,500,000 ($8.5 million) for fiscal cycle 2026.", ["$8,500,000", "fiscal cycle 2026"]),
        create_query("q-011", "sat-telemetry-spec-b2", "According to the component table, what is the weight and operating temperature of the High-Gain Antenna?", "table_lookup", [2], "High-Gain Antenna | 100 W | 18.5 kg | -50C to +85C", "The High-Gain Antenna is 18.5 kg and -50C to +85C.", ["18.5 kg", "-50C to +85C"], sources=["table", "text"], difficulty="medium"),
        create_query("q-012", "sat-telemetry-spec-b2", "Compare the power consumption of SAT-REF-77 during active downlink versus solar occultation standby mode across the specification.", "multi_page_reasoning", [2], "Nominal operating power consumption is 320 Watts during downlink and 45 Watts during solar occultation standby mode.", "Nominal power consumption is 320 Watts during active downlink and drops to 45 Watts during solar occultation standby mode.", ["320 Watts during downlink", "45 Watts during solar occultation"], difficulty="medium"),
        create_query("q-013", "sat-telemetry-spec-b2", "What is the laser optical communication downlink data rate for SAT-REF-77?", "negative_unanswerable", [], "", "This information is not provided in the specification. The document only specifies radio frequency Ka-band downlink, not optical laser communication.", [], is_answerable=False),
        create_query("q-014", "financial-quarterly-q3", "What was the total Q3 revenue reported by NovaTech Holdings and what was its year-over-year growth rate?", "factoid_text", [1], "NovaTech Holdings reported total revenue of $89.4 million for the third quarter of 2026, representing a 14.5% year-over-year growth.", "Total Q3 revenue was $89.4 million, representing a 14.5% year-over-year growth.", ["$89.4 million", "14.5% year-over-year growth"]),
        create_query("q-015", "financial-quarterly-q3", "What was the revenue, YoY growth, and operating margin of the Cloud Platform division according to the Q3 revenue table?", "table_lookup", [1], "Cloud Platform | $45.2M | +28.4% | 34.2%", "Cloud Platform revenue $45.2M, +28.4% YoY, 34.2% margin.", ["$45.2M", "+28.4%", "34.2%"], sources=["table", "text"], difficulty="medium"),
        create_query("q-016", "financial-quarterly-q3", "How much revenue did North America and EMEA contribute respectively in Q3?", "factoid_text", [2], "North America contributed $52.6 million (58.8% of total revenue), EMEA contributed $24.1 million (27.0%)", "North America contributed $52.6 million (58.8%) and EMEA contributed $24.1 million (27.0%).", ["$52.6 million", "58.8%", "$24.1 million", "27.0%"]),
        create_query("q-017", "financial-quarterly-q3", "Synthesize the total cash reserves and the specific capital expenditure amount spent on GPU datacenters in Q3.", "multi_page_reasoning", [3], "Capital expenditures in Q3 were $7.8 million, dedicated to GPU datacenter expansion in Northern Virginia. Total cash and cash equivalents stood at $142.5 million as of September 30, 2026.", "Capital expenditures were $7.8 million for GPU datacenter expansion in Northern Virginia, and total cash reserves stood at $142.5 million.", ["$7.8 million", "$142.5 million", "GPU datacenter expansion in Northern Virginia"], difficulty="medium"),
        create_query("q-018", "financial-quarterly-q3", "What was the stock dividend payout per share declared by NovaTech in Q3 2026?", "negative_unanswerable", [], "", "This information is not present in the Q3 financial report. No dividend payout is mentioned.", [], is_answerable=False),
    ]


def get_queries_part3() -> list[dict[str, Any]]:
    return [
        create_query("q-019", "database-engine-manual", "When are MemTables flushed to Level-0 SSTables in HyperStore DB, and how many Level-0 files trigger compaction?", "factoid_text", [1], "MemTables are flushed to Level-0 SSTables when active buffer size reaches 64 MB. Level-0 allows up to 4 overlapping files before triggering Level-0 to Level-1 compaction.", "MemTables are flushed when the active buffer size reaches 64 MB. Up to 4 overlapping files in Level-0 trigger compaction to Level-1.", ["64 MB", "4 overlapping files"]),
        create_query("q-020", "database-engine-manual", "What is the size amplification multiplier between levels, and what are the capacities of Level-1 and Level-2?", "factoid_text", [1], "Each level from Level-1 upward uses a 10x size amplification multiplier. Level-1 capacity is capped at 100 MB, Level-2 at 1,000 MB (1 GB)", "The size amplification multiplier is 10x. Level-1 capacity is capped at 100 MB and Level-2 at 1,000 MB (1 GB).", ["10x size amplification multiplier", "100 MB", "1,000 MB"]),
        create_query("q-021", "database-engine-manual", "What is the Write-Ahead Log (WAL) latency SLA and how many concurrent writes are coalesced per fsync cycle?", "factoid_text", [2], "Under the strict fsync policy, WAL write latency SLA is strictly guaranteed under 5.0 milliseconds. Group commit batching coalesces up to 32 concurrent writes per fsync cycle.", "The WAL write latency SLA is under 5.0 milliseconds, and group commit coalesces up to 32 concurrent writes per fsync cycle.", ["under 5.0 milliseconds", "up to 32 concurrent writes"]),
        create_query("q-022", "database-engine-manual", "What are the configuration parameters and target false positive rate for Bloom filter indexing in HyperStore DB?", "factoid_text", [2], "SSTable blocks include Block-level Bloom filters configured with 10 bits per key, achieving a target false positive rate of 1.0%.", "Bloom filters are configured with 10 bits per key, achieving a target false positive rate of 1.0%.", ["10 bits per key", "1.0%"]),
        create_query("q-023", "database-engine-manual", "Explain how HyperStore DB coordinates write durability and read point lookups across pages 1 and 2.", "multi_page_reasoning", [1, 2], "Writes append to WAL with <5ms latency and buffer into 64MB MemTables before flushing to Level-0. Point lookups utilize 10 bits/key Bloom filters (1% false positive) on SSTables across compacted levels.", "Writes achieve durability via synchronous WAL appends (<5ms SLA) and 64MB MemTable buffers flushed into an LSM-tree hierarchy. Point lookups are accelerated by Bloom filters (10 bits/key, 1% false positive rate) to bypass disk seeks across SSTable levels.", ["WAL write latency < 5.0ms", "64 MB MemTables", "Bloom filters 10 bits per key", "LSM storage engine"], difficulty="hard"),
        create_query("q-024", "database-engine-manual", "What is the B-link tree page split algorithm used by HyperStore DB?", "negative_unanswerable", [], "", "This information is not present in the manual. HyperStore DB uses an LSM-tree architecture, not B-link trees.", [], is_answerable=False),
        create_query("q-025", "clinical-trial-protocol", "What is the title and study design of clinical protocol MV-804?", "factoid_text", [1], "Protocol MV-804 is a randomized, double-blind, placebo-controlled Phase II trial evaluating the efficacy and safety of drug candidate MV-804 in adult patients diagnosed with Type 2 Diabetes Mellitus", "Protocol MV-804 is a randomized, double-blind, placebo-controlled Phase II clinical trial for Type 2 Diabetes Mellitus.", ["Protocol MV-804", "randomized, double-blind, placebo-controlled", "Phase II"]),
        create_query("q-026", "clinical-trial-protocol", "What are the specific age and baseline HbA1c inclusion criteria for Protocol MV-804?", "factoid_text", [1], "Inclusion criteria: Age between 18 and 65 years inclusive; baseline HbA1c between 7.5% and 10.0%", "Patients must be aged between 18 and 65 years inclusive, with baseline HbA1c between 7.5% and 10.0%.", ["18 and 65 years", "7.5% and 10.0%"]),
        create_query("q-027", "clinical-trial-protocol", "What are the three treatment arms and the total treatment duration in Protocol MV-804?", "factoid_text", [2], "Subjects are randomized 1:1:1 to receive oral MV-804 25 mg twice daily (BID), MV-804 50 mg twice daily (BID), or matching placebo for a total treatment duration of 24 weeks.", "The three arms are MV-804 25 mg BID, MV-804 50 mg BID, and matching placebo, administered for 24 weeks.", ["25 mg twice daily (BID)", "50 mg twice daily (BID)", "matching placebo", "24 weeks"], difficulty="medium"),
    ]


def get_queries_part4() -> list[dict[str, Any]]:
    return [
        create_query("q-028", "clinical-trial-protocol", "What is the primary efficacy endpoint specified in the MV-804 trial protocol?", "factoid_text", [2], "The primary efficacy endpoint is the mean change in HbA1c from baseline to Week 24.", "The primary efficacy endpoint is the mean change in HbA1c from baseline to Week 24.", ["mean change in HbA1c", "baseline to Week 24"]),
        create_query("q-029", "clinical-trial-protocol", "What is the required intravenous infusion rate for MV-804 in pediatric patients under 12?", "negative_unanswerable", [], "", "This information is not present. The protocol is an oral medication trial restricted to adults aged 18 to 65.", [], is_answerable=False),
        create_query("q-030", "cloud-security-compliance", "What are the password length, complexity, and rotation requirements under the ApexCloud IAM policy?", "factoid_text", [1], "Passwords must be a minimum of 16 characters in length, include uppercase, lowercase, numbers, and special symbols, and must be rotated every 90 days.", "Passwords must be at least 16 characters long, contain uppercase, lowercase, numbers, and symbols, and be rotated every 90 days.", ["minimum of 16 characters", "rotated every 90 days", "MFA"]),
        create_query("q-031", "cloud-security-compliance", "What encryption standard is enforced for data at rest and which TLS version is required for data in transit?", "factoid_text", [1], "All data at rest is encrypted using AES-256-GCM encryption with envelope keys managed by AWS KMS or HashiCorp Vault. Data in transit requires TLS 1.3 encryption", "Data at rest is encrypted with AES-256-GCM, and data in transit requires TLS 1.3.", ["AES-256-GCM", "TLS 1.3"]),
        create_query("q-032", "cloud-security-compliance", "How long must security audit logs be retained according to ApexCloud compliance standards?", "factoid_text", [2], "Security audit logs must be retained for a mandatory minimum period of 7 years in compliance with SOC 2 Type II and FedRAMP standards.", "Security audit logs must be retained for a mandatory minimum of 7 years.", ["minimum period of 7 years", "SOC 2 Type II"]),
        create_query("q-033", "cloud-security-compliance", "What is the SLA for initial containment and executive briefing during a P1 Critical security incident?", "factoid_text", [2], "P1 Critical security incidents require an initial containment response within 15 minutes of detection and executive briefing within 1 hour.", "Initial containment response is required within 15 minutes, and executive briefing within 1 hour.", ["within 15 minutes", "within 1 hour"]),
        create_query("q-034", "cloud-security-compliance", "What is the biological biometric iris scanning policy for data center entry under ApexCloud standard?", "negative_unanswerable", [], "", "This information is not provided in the security standard. Biometric iris scanning is not mentioned.", [], is_answerable=False),
        create_query("q-035", "network-switch-specs", "What is the total switching capacity and forwarding rate of the NetCore Nexus-9000 switch?", "factoid_text", [1], "It delivers a total non-blocking switching capacity of 6.4 Tbps and a packet forwarding rate of 4.7 Bpps.", "The switching capacity is 6.4 Tbps (Full Duplex) and the forwarding rate is 4.7 Bpps (billion packets per second).", ["6.4 Tbps", "4.7 Bpps"]),
        create_query("q-036", "network-switch-specs", "What port configuration and cut-through latency are listed in the Nexus-9000 parameter table?", "table_lookup", [1], "Port Configuration | 48x 100G QSFP28 + 8x 400G QSFP-DD\nLatency | 450 nanoseconds (cut-through)", "The port configuration is 48x 100G QSFP28 plus 8x 400G QSFP-DD, and cut-through latency is 450 nanoseconds.", ["48x 100G QSFP28", "8x 400G QSFP-DD", "450 nanoseconds"], sources=["table", "text"], difficulty="medium"),
    ]


def get_queries_part5() -> list[dict[str, Any]]:
    return [
        create_query("q-037", "network-switch-specs", "What is the maximum power consumption and typical idle power draw of the Nexus-9000?", "factoid_text", [2], "Maximum power consumption under full 100% traffic load is 450 Watts, with a typical idle power draw of 185 Watts.", "Maximum power consumption is 450 Watts under full load, and typical idle draw is 185 Watts.", ["450 Watts", "185 Watts"]),
        create_query("q-038", "network-switch-specs", "What is the Mean Time Between Failures (MTBF) rating for the Nexus-9000 switch chassis?", "factoid_text", [2], "Mean Time Between Failures (MTBF) is rated at 350,000 operating hours at 25C ambient temperature.", "The MTBF is rated at 350,000 operating hours at 25C ambient temperature.", ["350,000 operating hours"]),
        create_query("q-039", "network-switch-specs", "How many underwater submarine cable transceivers are integrated into the Nexus-9000 chassis?", "negative_unanswerable", [], "", "This information is not present in the datasheet. The Nexus-9000 is a data center switch, not a submarine cable terminal.", [], is_answerable=False),
        create_query("q-040", "solar-energy-research", "What solar irradiance model and baseline environmental parameters were used in the heliostat study?", "factoid_text", [1], "under the standard solar irradiance model AM1.5D (Direct solar irradiance 1000 W/m2 at 25C).", "The study used the AM1.5D model with direct solar irradiance of 1000 W/m2 at 25C.", ["AM1.5D", "1000 W/m2 at 25C"]),
        create_query("q-041", "solar-energy-research", "What peak thermal conversion efficiency and operating temperature was achieved by the molten salt central receiver?", "factoid_text", [1], "The molten salt central receiver attained a peak thermal conversion efficiency of 41.2% at a working temperature of 650C.", "The receiver achieved a peak thermal conversion efficiency of 41.2% at a working temperature of 650C.", ["41.2%", "650C"]),
        create_query("q-042", "solar-energy-research", "How many hours of full-load generation buffer does the two-tank molten salt thermal storage system provide?", "factoid_text", [2], "The two-tank indirect molten salt thermal storage system provides up to 8.0 hours of full-load turbine generation buffer", "The system provides up to 8.0 hours of full-load turbine generation buffer.", ["up to 8.0 hours"]),
        create_query("q-043", "solar-energy-research", "Explain the daily power dispatch profile and overall plant capacity factor increase illustrated in Figure 2.", "figure_chart_analysis", [2], "Figure 2: Daily Power Output Profile with Thermal Storage: The chart illustrates baseline solar generation peaking at 13:00 (120 MWth) followed by molten salt dispatch sustaining 85 MWe electrical output from 17:00 to 01:00. Overall plant capacity factor increased from 27.5% to 58.4%.", "Figure 2 shows solar generation peaking at 13:00 (120 MWth) with thermal storage sustaining 85 MWe electrical output from 17:00 to 01:00, raising the plant capacity factor from 27.5% to 58.4%.", ["Figure 2", "120 MWth at 13:00", "85 MWe output", "capacity factor increased from 27.5% to 58.4%"], sources=["text", "visual", "figure"], difficulty="medium"),
        create_query("q-044", "solar-energy-research", "What is the nuclear fusion plasma density measured inside the solar receiver?", "negative_unanswerable", [], "", "This information is not present. The research paper is about solar thermal power, not nuclear fusion.", [], is_answerable=False),
        create_query("q-045", "distributed-consensus-raft", "What is the randomized follower election timeout window in Raft, and what event occurs if it expires?", "factoid_text", [1], "follower election timeouts are randomized between 150 ms and 300 ms to prevent split-vote scenarios. If a follower receives no heartbeat AppendEntries RPC within this window, it transitions to Candidate state and increments the current term.", "Follower election timeouts are randomized between 150 ms and 300 ms. If no heartbeat is received, the follower transitions to Candidate state and increments the current term.", ["150 ms and 300 ms", "transitions to Candidate state"]),
    ]


def get_queries_part6() -> list[dict[str, Any]]:
    return [
        create_query("q-046", "distributed-consensus-raft", "How often does the cluster leader broadcast heartbeat AppendEntries RPCs?", "factoid_text", [1], "The cluster leader broadcasts empty AppendEntries RPC heartbeats every 50 ms to maintain leadership authority", "The cluster leader broadcasts heartbeats every 50 ms.", ["every 50 ms"]),
        create_query("q-047", "distributed-consensus-raft", "What is the strict majority quorum formula in Raft and how many acknowledgments are needed in a 5-node cluster?", "factoid_text", [2], "requires a strict majority quorum formula of (N/2) + 1 nodes. In a 5-node cluster, at least 3 nodes must acknowledge a log entry", "The quorum formula is (N/2) + 1. In a 5-node cluster, at least 3 nodes must acknowledge a log entry.", ["(N/2) + 1", "at least 3 nodes"]),
        create_query("q-048", "distributed-consensus-raft", "When is state machine snapshotting triggered in the Raft consensus engine?", "factoid_text", [2], "When log entries exceed 10,000 records, the Raft engine triggers asynchronous state machine snapshotting", "Snapshotting is triggered when log entries exceed 10,000 records.", ["exceed 10,000 records"]),
        create_query("q-049", "distributed-consensus-raft", "What is the Byzantine fault tolerance cryptographic signature scheme used in Raft?", "negative_unanswerable", [], "", "This information is not present. Raft is a crash fault-tolerant protocol, not a Byzantine fault-tolerant protocol.", [], is_answerable=False),
        create_query("q-050", "microservices-incident-postmortem", "What was the date, duration, and dropped request impact of Incident INC-4409?", "factoid_text", [1], "On August 14, 2026, Incident INC-4409 resulted in a 42-minute partial service degradation across the customer checkout gateway. Approximately 14,200 payment verification requests were dropped with HTTP 504 gateway timeouts.", "Incident INC-4409 occurred on August 14, 2026, lasted 42 minutes, and resulted in approximately 14,200 dropped payment requests.", ["August 14, 2026", "42-minute", "14,200 payment verification requests"]),
        create_query("q-051", "microservices-incident-postmortem", "What was the identified root cause of connection pool exhaustion during Incident INC-4409?", "factoid_text", [1], "The root cause was connection pool exhaustion in AuthService due to an unindexed database query on the user_sessions table during peak token refresh bursts. Maximum pool limit of 50 connections was saturated within 3 minutes.", "The root cause was an unindexed query on the user_sessions table in AuthService during peak token refresh bursts, which saturated the 50-connection pool.", ["unindexed database query on the user_sessions table", "AuthService", "50 connections"]),
        create_query("q-052", "microservices-incident-postmortem", "What composite index was added to resolve INC-4409 and how much did it reduce query latency?", "factoid_text", [2], "Added composite index on user_sessions(user_id, expires_at), reducing query latency from 850ms to 4ms", "A composite index on user_sessions(user_id, expires_at) was added, reducing query latency from 850ms to 4ms.", ["user_sessions(user_id, expires_at)", "850ms to 4ms"]),
        create_query("q-053", "microservices-incident-postmortem", "What was the complete timeline from first alert to full recovery during Incident INC-4409?", "factoid_text", [2], "14:02 UTC - First alert triggered; 14:12 UTC - Incident Commander engaged; 14:35 UTC - Hotfix index applied; 14:44 UTC - Full traffic recovery confirmed.", "The timeline was: 14:02 UTC first alert, 14:12 UTC Incident Commander engaged, 14:35 UTC hotfix index applied, and 14:44 UTC full recovery confirmed.", ["14:02 UTC", "14:12 UTC", "14:35 UTC", "14:44 UTC"], difficulty="medium"),
        create_query("q-054", "microservices-incident-postmortem", "What was the DDoS ransom demand requested by the attackers during INC-4409?", "negative_unanswerable", [], "", "This information is not present. Incident INC-4409 was an internal database performance issue, not an external DDoS or ransom attack.", [], is_answerable=False),
        create_query("q-055", "financial-quarterly-q3", "Synthesize the operational risks and Cloud customer retention rates across the NovaTech financial report.", "methodology_summary", [3], "Supply chain constraints in specialized semiconductor packaging remain the primary operational risk. Cloud customer retention rate remained strong at 96.4% across tier-1 enterprise contracts.", "The primary operational risk is specialized semiconductor packaging supply chain constraints, while Cloud customer retention remains solid at 96.4% across tier-1 enterprise contracts.", ["semiconductor packaging", "retention rate remained strong at 96.4%"], difficulty="medium"),
    ]


def get_all_gold_queries() -> list[dict[str, Any]]:
    return (
        get_queries_part1()
        + get_queries_part2()
        + get_queries_part3()
        + get_queries_part4()
        + get_queries_part5()
        + get_queries_part6()
    )


def write_gold_dataset(output_path: Path | str | None = None) -> Path:
    out = Path(output_path or DATASET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    queries = get_all_gold_queries()
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# DocuLens AI Official Gold Benchmark Dataset (v1.0 Release Freeze)\n")
        fh.write(f"# Total Gold Queries: {len(queries)} across 10 domain documents\n")
        fh.write("# Standard: Traceable ground-truth pages, answers, and category taxonomy\n")
        for q in queries:
            fh.write(json.dumps(q) + "\n")
    return out

