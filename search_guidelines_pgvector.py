import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
import json
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
import requests
import time
import cohere

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Cohere Embeddings Class
class CohereEmbeddings:
    def __init__(self, model: str = "embed-english-v3.0", api_key: str = None):
        """
        Initialize Cohere embeddings client
        
        Available models:
        - embed-english-v3.0 (1024 dimensions)
        - embed-multilingual-v3.0 (1024 dimensions)
        - embed-english-light-v3.0 (384 dimensions)
        - embed-multilingual-light-v3.0 (384 dimensions)
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
            
        self.client = cohere.Client(self.api_key)
        self.model = model
        self.input_type = "search_query"  # For query embeddings
        logging.info(f"Using Cohere model: {model}")
        
    def embed_query(self, text: str) -> List[float]:
        """Get embeddings from Cohere API"""
        try:
            response = self.client.embed(
                texts=[text],
                model=self.model,
                input_type=self.input_type,
                truncate="END"  # Truncate if text is too long
            )
            
            # Return the first (and only) embedding
            return response.embeddings[0]
            
        except cohere.errors.TooManyRequestsError:
            logging.warning("Rate limit hit, waiting before retry...")
            time.sleep(60)  # Wait 60 seconds before retry
            return self.embed_query(text)
            
        except Exception as e:
            error_msg = f"Cohere API error: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)

# Initialize embeddings
embeddings = None

# Try Cohere
try:
    logging.info("Initializing Cohere embedding client...")
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",  # 1024 dimensions, same as mxbai-embed-large-v1
        api_key=os.getenv("COHERE_API_KEY")
    )
    logging.info("Cohere client initialized successfully")
    
    # Test the embeddings
    test_embedding = embeddings.embed_query("test")
    logging.info(f"Embedding dimension: {len(test_embedding)}")
    
except Exception as e:
    logging.error(f"Error initializing Cohere client: {str(e)}")
    

# Initialize Groq client for LLM analysis
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    groq_client = Groq(api_key=groq_api_key)
    logging.info("Groq client initialized successfully")
except Exception as e:
    logging.error(f"Error initializing Groq client: {str(e)}")
    groq_client = None

class NeonVectorSearch:
    """Custom vector search using Neon PostgreSQL with pgvector extension."""
    
    def __init__(self, connection_string: str, table_name: str = "clinical_guidelines"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish connection to database."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            register_vector(self.conn)
            logging.info("Connected to Neon database")
        except Exception as e:
            logging.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _reconnect_if_needed(self):
        """Reconnect if connection is closed."""
        if self.conn is None or self.conn.closed:
            self._connect()
    
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 5,
        filters: Dict[str, Any] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform similarity search with optional filters.
        
        Returns:
            List of tuples (document, distance)
        """
        self._reconnect_if_needed()
        
        # Start a new transaction
        try:
            # Rollback any existing failed transaction
            try:
                self.conn.rollback()
            except:
                pass
            
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause from filters
            where_conditions = []
            params = []
            
            if filters:
                for key, value in filters.items():
                    if value is not None and value != "":
                        if key == "is_distilled":
                            where_conditions.append(f"{key} = %s")
                            params.append(value)
                        else:
                            where_conditions.append(f"{key} = %s")
                            params.append(value)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Convert embedding to string format for PostgreSQL
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Query with cosine distance
            query = f"""
            SELECT 
                id,
                content,
                1 - (embedding <=> %s::vector) as similarity,
                embedding <=> %s::vector as distance,
                medical_domain,
                file_path,
                section,
                chunk_type,
                chunk_category,
                original_text,
                chunk_id,
                is_distilled,
                guideline_title,
                source_file,
                evidence_certainty,
                recommendation_strength,
                diabetes_type,
                technology_type,
                medication_type,
                outcome_type,
                condition_type,
                severity_classification,
                assessment_type,
                treatment_type,
                medication_class,
                scale_type,
                demographic_focus,
                population,
                metadata
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            # Execute with all parameters
            all_params = [embedding_str, embedding_str] + params + [embedding_str, k]
            cursor.execute(query, all_params)
            results = cursor.fetchall()
            
            # Commit the transaction
            self.conn.commit()
            
            # Format results similar to ChromaDB output
            formatted_results = []
            for row in results:
                # Create document-like object
                doc = {
                    "page_content": row["content"],
                    "metadata": {
                        "medical_domain": row["medical_domain"],
                        "file_path": row["file_path"],
                        "section": row["section"],
                        "chunk_type": row["chunk_type"],
                        "chunk_category": row["chunk_category"],
                        "original_text": row["original_text"],
                        "chunk_id": row["chunk_id"],
                        "is_distilled": row["is_distilled"],
                        "guideline_title": row["guideline_title"],
                        "source_file": row["source_file"],
                        "evidence_certainty": row["evidence_certainty"],
                        "recommendation_strength": row["recommendation_strength"],
                        "diabetes_type": row["diabetes_type"],
                        "technology_type": row["technology_type"],
                        "medication_type": row["medication_type"],
                        "outcome_type": row["outcome_type"],
                        "condition_type": row["condition_type"],
                        "severity_classification": row["severity_classification"],
                        "assessment_type": row["assessment_type"],
                        "treatment_type": row["treatment_type"],
                        "medication_class": row["medication_class"],
                        "scale_type": row["scale_type"],
                        "demographic_focus": row["demographic_focus"],
                        "population": row["population"]
                    }
                }
                
                # Add any additional metadata from JSON
                if row["metadata"]:
                    doc["metadata"].update(row["metadata"])
                
                formatted_results.append((doc, row["distance"]))
            
            cursor.close()
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            # Rollback on error
            try:
                self.conn.rollback()
            except:
                pass
            raise
    
    def close(self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logging.info("Database connection closed")

# Initialize Neon vector search
try:
    neon_connection_string = os.environ.get("NEON_DATABASE_URL")
    if not neon_connection_string:
        raise ValueError("NEON_DATABASE_URL environment variable not set")
    
    logging.info("Connecting to Neon pgvector database...")
    vector_search = NeonVectorSearch(
        connection_string=neon_connection_string,
        table_name="clinical_guidelines"
    )
    logging.info("Neon pgvector connection established successfully")
except Exception as e:
    logging.error(f"Error connecting to Neon pgvector: {str(e)}")
    raise

# [Rest of the code remains the same - search_guidelines, extract_quotes_from_chunk, analyze_with_llm, etc.]
def search_guidelines(
    query: str, 
    top_k: int = 5,
    diabetes_type: Optional[str] = None,
    population: Optional[str] = None, 
    technology_type: Optional[str] = None,
    medication_type: Optional[str] = None,
    outcome_type: Optional[str] = None,
    evidence_certainty: Optional[str] = None,
    recommendation_strength: Optional[str] = None,
    prioritize_distilled: bool = False
) -> List[Dict[str, Any]]:
    """
    Search the clinical guidelines database for clinically relevant content.
    
    Args:
        query: Natural language clinical question
        top_k: Number of results to return
        diabetes_type: Filter by "type_1" or "type_2"
        population: Filter by "children" or "adults"
        technology_type: Filter by "CGM", "CSII", "Flash", etc.
        medication_type: Filter by "SGLT2", "GLP1", "DPP4", "metformin", etc.
        outcome_type: Filter by "HbA1c", "hypoglycemia", "quality_of_life", etc.
        evidence_certainty: Filter by "high", "moderate", "low", "very_low"
        recommendation_strength: Filter by "strong" or "conditional"
        prioritize_distilled: Whether to prioritize atomic facts over original text
    
    Returns:
        List of formatted search results with clinical metadata
    """
    try:
        # Get embedding for the query
        query_embedding = embeddings.embed_query(query)
        
        # Build filters dictionary
        filters = {}
        
        if diabetes_type:
            filters["diabetes_type"] = diabetes_type
        if population:
            filters["population"] = population
        if technology_type:
            filters["technology_type"] = technology_type
        if medication_type:
            filters["medication_type"] = medication_type
        if outcome_type:
            filters["outcome_type"] = outcome_type
        if evidence_certainty:
            filters["evidence_certainty"] = evidence_certainty
        if recommendation_strength:
            filters["recommendation_strength"] = recommendation_strength
        if prioritize_distilled:
            filters["is_distilled"] = True
        
        # Perform search with filters
        try:
            if filters:
                results = vector_search.similarity_search_with_score(
                    query_embedding=query_embedding,
                    k=top_k * 2,  # Get more results to account for filtering
                    filters=filters
                )
                # Take only top_k after filtering
                results = results[:top_k]
            else:
                results = vector_search.similarity_search_with_score(
                    query_embedding=query_embedding,
                    k=top_k
                )
            
            # If similarity scores are very low and we were filtering by distilled content, retry without filter
            if results and prioritize_distilled and all(distance > 0.9 for _, distance in results):
                logging.info("Low similarity with distilled filter, retrying without distilled filter...")
                
                # Search again without distilled filter
                filters_without_distilled = {k: v for k, v in filters.items() if k != "is_distilled"}
                retry_results = vector_search.similarity_search_with_score(
                    query_embedding=query_embedding,
                    k=top_k,
                    filters=filters_without_distilled if filters_without_distilled else None
                )
                
                # Use retry results if they're better
                if retry_results and any(distance < 0.9 for _, distance in retry_results):
                    results = retry_results
                    logging.info(f"Using non-filtered results: {len(results)} found")
                    
        except Exception as filter_error:
            logging.warning(f"Filter search failed: {filter_error}. Falling back to unfiltered search.")
            # Fallback to unfiltered search
            results = vector_search.similarity_search_with_score(
                query_embedding=query_embedding,
                k=top_k
            )
        
        # Format results with clinical context
        formatted_results = []
        for i, (doc, distance) in enumerate(results, 1):
            metadata = doc["metadata"]
            
            # Calculate similarity score (0-1, higher is better)
            similarity_score = max(0, 1 - distance)
            
            # Extract clinical context
            clinical_context = {
                "diabetes_type": metadata.get("diabetes_type") or "Not specified",
                "population": metadata.get("population") or "Not specified",
                "technology_type": metadata.get("technology_type") or "Not specified",
                "medication_type": metadata.get("medication_type") or "Not specified",
                "outcome_type": metadata.get("outcome_type") or "Not specified",
                "evidence_certainty": metadata.get("evidence_certainty") or "Not specified",
                "recommendation_strength": metadata.get("recommendation_strength") or "Not specified"
            }
            
            # Determine content source and display
            content_text = doc["page_content"]  # This contains the embedded atomic facts
            original_text = metadata.get("original_text", "")
            is_distilled = metadata.get("is_distilled", False)
            
            formatted_result = {
                "chunk_number": i,  # Add chunk number for citations
                "similarity_score": similarity_score,
                "section": metadata.get("section", "Unknown section"),
                "source_file": metadata.get("file_path", "Unknown file"),
                "chunk_id": metadata.get("chunk_id", ""),
                "content_type": "Atomic Facts" if is_distilled else "Original Text",
                "clinical_context": clinical_context,
                "embedded_content": content_text,  # The atomic facts that were embedded
                "original_context": original_text,  # The original guideline text for reference
                "guideline_title": metadata.get("guideline_title", ""),
                "source_reference": metadata.get("source_file", "")
            }
            
            formatted_results.append(formatted_result)
        
        # Sort by similarity score (highest first) and evidence certainty
        formatted_results.sort(key=lambda x: (
            x["similarity_score"],
            1 if x["clinical_context"]["evidence_certainty"] == "high" else
            0.8 if x["clinical_context"]["evidence_certainty"] == "moderate" else
            0.6 if x["clinical_context"]["evidence_certainty"] == "low" else
            0.4 if x["clinical_context"]["evidence_certainty"] == "very_low" else 0.2
        ), reverse=True)
        
        logging.info(f"Search completed: {len(formatted_results)} results found for query: '{query}'")
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        return []

def extract_quotes_from_chunk(content: str, max_quotes: int = 3) -> List[str]:
    """Extract relevant sentences from chunk content for quoting."""
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Filter out very short sentences and clean them
    quotes = []
    for sentence in sentences:
        cleaned = sentence.strip()
        if len(cleaned) > 20 and not cleaned.startswith('[') and not cleaned.startswith('('):
            quotes.append(cleaned)
            if len(quotes) >= max_quotes:
                break
    
    return quotes

def analyze_with_llm(query: str, search_results: List[Dict[str, Any]], model: str = "llama-3.3-70b-versatile") -> str:
    """
    Analyze search results using Groq LLM with strict citation requirements.
    
    Args:
        query: The user's question
        search_results: List of search results from the vector database
        model: Groq model to use
    
    Returns:
        LLM's analysis with citations
    """
    if not groq_client:
        return "LLM analysis unavailable - Groq client not initialized"
    
    if not search_results:
        return "The information to answer this question is not found in the provided documents"
    
    # Prepare chunks for the prompt
    chunks_text = ""
    all_quotes = {}
    
    for result in search_results:
        chunk_num = result['chunk_number']
        
        # Use original context if available, otherwise use embedded content
        content = result['original_context'] if result['original_context'] else result['embedded_content']
        
        # Extract potential quotes from this chunk
        quotes = extract_quotes_from_chunk(content, max_quotes=3)
        all_quotes[f"REF{chunk_num}"] = quotes  # Changed from CHUNK to REF
        
        chunks_text += f"\n[REF{chunk_num}]\n"  # Changed from CHUNK to REF
        chunks_text += f"Section: {result['section']}\n"
        chunks_text += f"Source: {result['source_file'].split('/')[-1]}\n"
        chunks_text += f"Content: {content}\n"
        chunks_text += "-" * 40
    
    # Create the analysis prompt
    prompt = f"""You are a medical information specialist analyzing clinical guidelines. Answer the following question using ONLY the provided document chunks.

QUESTION: {query}

PROVIDED DOCUMENT CHUNKS:
{chunks_text}

IMPORTANT INSTRUCTIONS:
1. ONLY use information from the provided documents
2. Be direct and concise in your answer
3. If the answer is NOT FOUND, PARTIALLY FOUND or NOT SURE in the documents, clearly state "The information to answer this question is not found in the provided documents" and do not give any quotes or reference information.
4. Do not make assumptions or add information not present in the documents
5. Focus on accuracy over completeness
7. EVERY statement must include a citation using [REF1], [REF2], etc.
8. YOU MUST include direct quotes from the chunks to support your statements
9. Format quotes as: "exact text from document" [REF1]
10. When paraphrasing, still cite the source: paraphrased information [REF2]

REQUIRED CITATION FORMAT:
- For direct quotes: "exact quote from the chunk" [REF1]
- For paraphrased info: summarized statement [REF2]
- Multiple sources: statement [REF1, REF3]

IMPORTANT: Include at least 2-3 direct quotes in your answer to support key points

Provide your analysis:"""

    try:
        # Make request to Groq API
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical information specialist. Always cite sources using [REF#] format and include direct quotes to support your statements. Never provide information not found in the provided chunks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=1000,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logging.error(f"Error in LLM analysis: {str(e)}")
        return f"Error generating analysis: {str(e)}"
    
def display_clinical_results(results: List[Dict[str, Any]], show_original: bool = True):
    """Display search results in a simple, clean format."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFOUND {len(results)} RESULTS:")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. SIMILARITY: {result['similarity_score']:.3f} | {result['section']}")
        
        # Show facts
        print(f"\nFACTS:")
        print(result['embedded_content'])
        
        # Show original text
        if show_original and result['original_context']:
            print(f"\nORIGINAL:")
            # Show first 300 chars of original text
            original = result['original_context'][:300]
            if len(result['original_context']) > 300:
                original += "..."
            print(original)
        
        print(f"\nFILE: {result['source_file'].split('\\')[-1].split('/')[-1]}")
        
        if i < len(results):
            print("-" * 60)

def display_llm_analysis(analysis: str):
    """Display the LLM analysis in a formatted way."""
    print("\n" + "="*60)
    print("🤖 LLM ANALYSIS")
    print("="*60)
    print(analysis)
    print("="*60)

def get_clinical_filters():
    """Interactive filter selection for clinical searches."""
    print("\n🔍 CLINICAL SEARCH FILTERS (Press Enter to skip any filter)")
    print("="*60)
    
    filters = {}
    
    # Diabetes type filter
    diabetes_type = input("Diabetes Type (type_1/type_2): ").strip().lower()
    if diabetes_type in ['type_1', 'type_2']:
        filters['diabetes_type'] = diabetes_type
    
    # Population filter  
    population = input("Population (children/adults): ").strip().lower()
    if population in ['children', 'adults']:
        filters['population'] = population
    
    # Technology filter
    technology = input("Technology (CGM/CSII/Flash/AutoCSII): ").strip().upper()
    if technology in ['CGM', 'CSII', 'FLASH', 'AUTOCSII']:
        filters['technology_type'] = technology
    
    # Medication filter
    medication = input("Medication (SGLT2/GLP1/DPP4/metformin): ").strip().upper()
    if medication in ['SGLT2', 'GLP1', 'DPP4'] or medication.lower() == 'metformin':
        filters['medication_type'] = medication if medication != 'metformin' else 'metformin'
    
    # Outcome filter
    outcome = input("Outcome (HbA1c/hypoglycemia/quality_of_life): ").strip().lower()
    if outcome in ['hba1c', 'hypoglycemia', 'quality_of_life']:
        filters['outcome_type'] = outcome
    
    # Evidence certainty filter
    certainty = input("Evidence Certainty (high/moderate/low/very_low): ").strip().lower()
    if certainty in ['high', 'moderate', 'low', 'very_low']:
        filters['evidence_certainty'] = certainty
    
    # Recommendation strength filter
    strength = input("Recommendation Strength (strong/conditional): ").strip().lower()
    if strength in ['strong', 'conditional']:
        filters['recommendation_strength'] = strength
    
    return filters

def show_example_queries():
    """Display example GP clinical queries."""
    examples = [
        "What are the first-line medications for adults with type 2 diabetes?",
        "How is atopic dermatitis severity classified using EASI scores?",
        "CGM recommendations for children with type 1 diabetes",
        "What is dupilumab and when is it used for atopic dermatitis?",
        "SGLT-2 inhibitor cardiovascular benefits",
        "Topical corticosteroid use in atopic dermatitis management",
        "Insulin pump therapy effectiveness in children", 
        "DLQI quality of life assessment in dermatitis patients",
        "Metformin adverse effects and contraindications",
        "When should atopic dermatitis patients be referred to specialists?",
        "GLP-1 agonist recommendations for weight management",
        "JAK inhibitors for severe atopic dermatitis treatment",
        "Continuous glucose monitoring accuracy and reliability",
        "Wet wrap therapy for moderate atopic dermatitis",
        "Cost effectiveness of diabetes technologies",
        "Assessment scales for atopic dermatitis severity"
    ]
    
    print("\n💡 EXAMPLE CLINICAL QUERIES (DIABETES & DERMATITIS):")
    print("="*60)
    for i, example in enumerate(examples, 1):
        print(f"{i:2}. {example}")
    print("="*60)

def main():
    """Enhanced search interface with LLM analysis."""
    print("\nAUSTRALIAN DIABETES AND DERMATITIS GUIDELINES SEARCH")
    print("="*60)
    print("Search modes:")
    print("1. Simple search - Type your question")
    print("2. Advanced search - Type 'advanced' for filtered search")
    print("3. Examples - Type 'examples' to see sample queries")
    print("4. Exit - Type 'quit' to exit")
    print("="*60)
    
    # Check if LLM is available
    if groq_client:
        print("✅ LLM Analysis: ENABLED (Groq)")
    else:
        print("❌ LLM Analysis: DISABLED (Set GROQ_API_KEY environment variable)")
    
    print("✅ Vector Database: Neon pgvector")
    
    while True:
        query = input("\nQuestion: ").strip()
        
        if query.lower() == 'quit':
            print("Goodbye!")
            break
        
        if query.lower() == 'examples':
            show_example_queries()
            continue
            
        if query.lower() == 'advanced':
            # Get filters from user
            filters = get_clinical_filters()
            query = input("\nEnter your question: ").strip()
            if not query:
                continue
            
            # Search with filters
            results = search_guidelines(
                query, 
                top_k=5,
                **filters,
                prioritize_distilled=False
            )
        else:
            # Simple search
            results = search_guidelines(query, top_k=5, prioritize_distilled=False)
        
        # Display search results
        display_clinical_results(results, show_original=True)
        
        # Perform LLM analysis if available
        if groq_client and results:
            print("\n⏳ Analyzing with LLM...")
            analysis = analyze_with_llm(query, results)
            display_llm_analysis(analysis)
        elif not groq_client:
            print("\n⚠️  LLM analysis not available - set GROQ_API_KEY to enable")
    
    # Clean up
    vector_search.close()

if __name__ == "__main__":
    main()