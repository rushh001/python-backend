import json
import os
import sys
from typing import Any

# --- Import your real search and LLM logic ---
# Add the parent directory to sys.path so we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Modified line
try:
    from search_guidelines_pgvector import search_guidelines, analyze_with_llm
except ImportError as e:
    print(f"[IMPORT ERROR] {e}", file=sys.stderr)
    # Fallback: dummy functions if import fails
    def search_guidelines(query, top_k=5, **kwargs):
        return [
            {"chunk_number": 1, "embedded_content": "Dummy evidence 1", "original_context": "", "section": "Section A", "source_file": "file1.txt"},
            {"chunk_number": 2, "embedded_content": "Dummy evidence 2", "original_context": "", "section": "Section B", "source_file": "file2.txt"}
        ]
    def analyze_with_llm(query, search_results, model=None):
        return f"LLM answer for: {query} (dummy)"


def handler(request: Any):
    # Parse input (POST with JSON body, or GET with query param)
    if request.method == "POST":
        try:
            body = request.body.decode("utf-8")
            data = json.loads(body)
        except Exception:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid JSON"}),
                "headers": {"Content-Type": "application/json"}
            }
    else:
        data = request.args

    query = data.get("query", "")
    # Optional: support filters from frontend
    filters = {k: v for k, v in data.items() if k != "query"}

    # --- Real pipeline: embed, search, LLM ---
    try:
        # 1. Vector search
        results = search_guidelines(query, top_k=5, **filters)
        # 2. LLM answer
        answer = analyze_with_llm(query, results)
        # 3. Format references for frontend
        references = [
            {
                "chunk_number": r.get("chunk_number"),
                "section": r.get("section"),
                "source_file": r.get("source_file"),
                "embedded_content": r.get("embedded_content"),
                "original_context": r.get("original_context")
            }
            for r in results
        ]
        return {
            "statusCode": 200,
            "body": json.dumps({
                "answer": answer,
                "references": references
            }),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }

if __name__ == "__main__":
    class DummyRequest:
        def __init__(self, method, body=None, args=None):
            self.method = method
            self.body = body or b''
            self.args = args or {}

    # Example POST request
    req = DummyRequest(
        method="POST",
        body=json.dumps({"query": "What are the first-line medications for type 2 diabetes?"}).encode("utf-8")
    )
    response = handler(req)
    print("Status:", response["statusCode"])
    print("Body:", response["body"])