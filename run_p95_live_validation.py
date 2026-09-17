#!/usr/bin/env python3
"""Phase 9.5 Live Docker System Validation Runner."""
import io, json, time, urllib.request, urllib.error, pymupdf

API_BASE = "http://localhost:8000/api/v1"
FRONTEND_BASE = "http://localhost:3000"

def make_sample_pdf() -> bytes:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "DocuLens AI Architecture and Reliability Report", fontsize=16)
    p1.insert_text((50, 100), "DocuLens AI delivers multimodal document intelligence using hybrid BM25 and dense retrieval.", fontsize=11)
    p1.insert_text((50, 120), "The fiscal year 2025 operating budget was exactly 42.5 million dollars.", fontsize=11)
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Section 2: Performance and Security", fontsize=14)
    p2.insert_text((50, 100), "Vector store indexing latency averages under 15 milliseconds per chunk.", fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def run_live():
    res = {}
    # 1. Health & Readiness
    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{API_BASE}/health") as r:
        res["health_body"] = json.loads(r.read().decode())
    res["health_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    with urllib.request.urlopen(f"{API_BASE}/health/ready") as r:
        res["ready_body"] = json.loads(r.read().decode())
    res["ready_ms"] = (time.perf_counter() - t0) * 1000

    # 2. Upload valid PDF
    pdf_data = make_sample_pdf()
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="sample.pdf"\r\nContent-Type: application/pdf\r\n\r\n')
    body.write(pdf_data)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{API_BASE}/documents/upload",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Request-ID": "val-req-001"}
    )
    with urllib.request.urlopen(req) as r:
        upload_resp = json.loads(r.read().decode())
        upload_req_id = r.headers.get("x-request-id")
    res["upload_ms"] = (time.perf_counter() - t0) * 1000
    res["doc_id"] = upload_resp["document_id"]
    res["upload_req_id"] = upload_req_id

    # 3. Index document
    t0 = time.perf_counter()
    req_idx = urllib.request.Request(f"{API_BASE}/documents/{res['doc_id']}/index", data=b"")
    with urllib.request.urlopen(req_idx) as r:
        idx_resp = json.loads(r.read().decode())
    res["index_ms"] = (time.perf_counter() - t0) * 1000
    res["index_body"] = idx_resp

    # 4. Grounded QA
    q = json.dumps({"question": "What was the operating budget in fiscal year 2025?"}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{API_BASE}/documents/{res['doc_id']}/ask",
        data=q,
        headers={"Content-Type": "application/json", "X-Request-ID": "val-ask-001"}
    )
    with urllib.request.urlopen(req) as r:
        ask_resp = json.loads(r.read().decode())
        ask_req_id = r.headers.get("x-request-id")
    res["ask_total_ms"] = (time.perf_counter() - t0) * 1000
    res["ask_body"] = ask_resp
    res["ask_req_id"] = ask_req_id

    # 5. Refusal Query
    q_refuse = json.dumps({"question": "What is the atmospheric composition of Neptune in 1600?"}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{API_BASE}/documents/{res['doc_id']}/ask",
        data=q_refuse,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        refuse_resp = json.loads(r.read().decode())
    res["refuse_ms"] = (time.perf_counter() - t0) * 1000
    res["refuse_body"] = refuse_resp

    # 6. Failure Cases (400 Corrupt & 404 Missing)
    bad_req = urllib.request.Request(
        f"{API_BASE}/documents/upload",
        data=b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"bad.pdf\"\r\nContent-Type: application/pdf\r\n\r\nCORRUPT\r\n--b--\r\n",
        headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    try:
        urllib.request.urlopen(bad_req)
    except urllib.error.HTTPError as e:
        res["corrupt_upload_status"] = e.code

    missing_req = urllib.request.Request(f"{API_BASE}/documents/doc_000000000000/ask", data=q, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(missing_req)
    except urllib.error.HTTPError as e:
        res["missing_doc_status"] = e.code

    # 7. Frontend
    with urllib.request.urlopen(FRONTEND_BASE) as r:
        res["frontend_status"] = r.status

    with open("/home/keshav/Sid Workspace/Projects/doculens-ai/validation_results_phase95.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Done! Recorded live metrics in validation_results_phase95.json")

if __name__ == "__main__":
    run_live()
