from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import uvicorn
from datetime import datetime

# Import your existing modules
from search_guidelines_pgvector import search_guidelines, analyze_with_llm

load_dotenv()

app = FastAPI(title="Clinical Guidelines RAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://nextjs-frontend-4mr7.vercel.app/",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    diabetes_type: Optional[str] = None
    population: Optional[str] = None
    technology_type: Optional[str] = None
    medication_type: Optional[str] = None
    outcome_type: Optional[str] = None
    evidence_certainty: Optional[str] = None
    recommendation_strength: Optional[str] = None
    prioritize_distilled: Optional[bool] = False
    top_k: Optional[int] = 5

class Reference(BaseModel):
    chunk_number: int
    section: str
    source_file: str
    embedded_content: str
    original_context: str
    similarity_score: Optional[float] = None
    clinical_context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    references: List[Reference]
    success: bool = True
    error: Optional[str] = None

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main endpoint for RAG queries."""
    try:
        # Call your existing search function
        results = search_guidelines(
            query=request.query,
            top_k=request.top_k,
            diabetes_type=request.diabetes_type,
            population=request.population,
            technology_type=request.technology_type,
            medication_type=request.medication_type,
            outcome_type=request.outcome_type,
            evidence_certainty=request.evidence_certainty,
            recommendation_strength=request.recommendation_strength,
            prioritize_distilled=request.prioritize_distilled
        )
        
        # Generate answer using your existing LLM function
        answer = analyze_with_llm(request.query, results)
        
        # Format references
        references = []
        for result in results:
            ref = Reference(
                chunk_number=result.get("chunk_number", 0),
                section=result.get("section", ""),
                source_file=result.get("source_file", ""),
                embedded_content=result.get("embedded_content", ""),
                original_context=result.get("original_context", ""),
                similarity_score=result.get("similarity_score"),
                clinical_context=result.get("clinical_context", {})
            )
            references.append(ref)
        
        return QueryResponse(
            answer=answer,
            references=references,
            success=True
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return QueryResponse(
            answer="",
            references=[],
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint that returns immediately."""
    return {"status": "healthy", "service": "Clinical Guidelines RAG API", "timestamp": str(datetime.now())}

# Also add a readiness check
@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Quick check without heavy operations
        return {"status": "ready", "timestamp": str(datetime.now())}
    except Exception as e:
        return {"status": "not ready", "error": str(e)}, 503

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clinical Guidelines RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)