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
        "https://*.vercel.app",
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
    
    # Initialize in background to not block startup
    asyncio.create_task(initialize_resources())

async def initialize_resources():
    """Initialize heavy resources in background"""
    global search_guidelines, analyze_with_llm
    
    try:
        # Small delay to ensure server is fully up
        await asyncio.sleep(2)
        
        logger.info("Starting resource initialization...")
        
        # Initialize app state
        if app_state.initialize():
            # Now import the functions
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

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main endpoint for RAG queries."""
    
    # Check if resources are loaded
    if search_guidelines is None or analyze_with_llm is None:
        return QueryResponse(
            answer="",
            references=[],
            success=False,
            error="Service is still initializing. Please try again in a few seconds."
        )
    
    try:
        # Your existing query logic
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
        
        answer = analyze_with_llm(request.query, results)
        
        # Format references
        references = []
        for result in results:
            references.append({
                "chunk_number": result.get("chunk_number", 0),
                "section": result.get("section", ""),
                "source_file": result.get("source_file", ""),
                "embedded_content": result.get("embedded_content", ""),
                "original_context": result.get("original_context", ""),
                "similarity_score": result.get("similarity_score"),
                "clinical_context": result.get("clinical_context", {})
            })
        
        return QueryResponse(
            answer=answer,
            references=references,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return QueryResponse(
            answer="",
            references=[],
            success=False,
            error=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Clinical Guidelines RAG API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "health": "/health",
            "ready": "/ready"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)