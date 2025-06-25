from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import uvicorn
import logging
import asyncio
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Import app_state but don't initialize heavy resources yet
from app_state import app_state

# Don't import search_guidelines_pgvector at module level
search_guidelines = None
analyze_with_llm = None

# Request/Response Models
class SearchRequest(BaseModel):
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

class SearchResult(BaseModel):
    chunk_number: int
    section: str
    source_file: str
    embedded_content: str
    original_context: str
    similarity_score: Optional[float] = None
    clinical_context: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    success: bool = True
    error: Optional[str] = None

class GenerateAnswerRequest(BaseModel):
    query: str
    search_results: List[Dict[str, Any]]  # The results from the search API

class GenerateAnswerResponse(BaseModel):
    answer: str
    success: bool = True
    error: Optional[str] = None

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

class QueryResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize resources after server starts"""
    logger.info("FastAPI server started, initializing resources in background...")
    asyncio.create_task(initialize_resources())

async def initialize_resources():
    """Initialize heavy resources in background"""
    global search_guidelines, analyze_with_llm
    
    try:
        await asyncio.sleep(2)
        logger.info("Starting resource initialization...")
        
        if app_state.initialize():
            from search_guidelines_pgvector import search_guidelines as sg, analyze_with_llm as awl
            search_guidelines = sg
            analyze_with_llm = awl
            logger.info("All resources loaded successfully")
        else:
            logger.error("Failed to initialize app state")
            
    except Exception as e:
        logger.error(f"Error initializing resources: {e}")

@app.get("/health")
async def health_check():
    """Simple health check that doesn't require resources"""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "Clinical Guidelines RAG API"}
    )

@app.get("/ready")
async def readiness_check():
    """Check if the app is ready to handle requests"""
    if search_guidelines is None or analyze_with_llm is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "message": "Resources still initializing"}
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready"}
    )

# API 1: Vector Search Endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """Search for relevant clinical guidelines using vector similarity."""
    
    if search_guidelines is None:
        return SearchResponse(
            results=[],
            query=request.query,
            success=False,
            error="Service is still initializing. Please try again in a few seconds."
        )
    
    try:
        # Perform vector search
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
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(SearchResult(
                chunk_number=result.get("chunk_number", 0),
                section=result.get("section", ""),
                source_file=result.get("source_file", ""),
                embedded_content=result.get("embedded_content", ""),
                original_context=result.get("original_context", ""),
                similarity_score=result.get("similarity_score"),
                clinical_context=result.get("clinical_context", {})
            ))
        
        return SearchResponse(
            results=formatted_results,
            query=request.query,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return SearchResponse(
            results=[],
            query=request.query,
            success=False,
            error=str(e)
        )

# API 2: Generate Answer Endpoint
@app.post("/api/generate-answer", response_model=GenerateAnswerResponse)
async def generate_answer_endpoint(request: GenerateAnswerRequest):
    """Generate a polished answer from search results using LLM."""
    
    if analyze_with_llm is None:
        return GenerateAnswerResponse(
            answer="",
            success=False,
            error="Service is still initializing. Please try again in a few seconds."
        )
    
    try:
        # Generate answer using LLM
        answer = analyze_with_llm(request.query, request.search_results)
        
        return GenerateAnswerResponse(
            answer=answer,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return GenerateAnswerResponse(
            answer="",
            success=False,
            error=str(e)
        )



@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Clinical Guidelines RAG API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/api/search - Search for relevant guidelines",
            "generate": "/api/generate-answer - Generate answer from search results",
            "health": "/health - Health check",
            "ready": "/ready - Readiness check"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)