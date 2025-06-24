import os
import json
import glob
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Initialize Ollama embeddings
try:
    logging.info("Initializing Ollama client...")
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    logging.info("Ollama client initialized successfully")
except Exception as e:
    logging.error(f"Error initializing Ollama: {str(e)}")
    raise

class NeonPgVectorStore:
    """Custom vector store using Neon PostgreSQL with pgvector extension."""
    
    def __init__(self, connection_string: str, table_name: str = "clinical_guidelines", embedding_dimension: int = 1024):
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.conn = None
        self.cursor = None
        self._connect()
        self._setup_database()
    
    def _connect(self):
        """Establish connection to Neon database."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.cursor = self.conn.cursor()
            # Register pgvector extension
            register_vector(self.conn)
            logging.info("Connected to Neon database successfully")
        except Exception as e:
            logging.error(f"Error connecting to Neon database: {str(e)}")
            raise
    
    def _setup_database(self):
        """Create necessary tables and extensions."""
        try:
            # Create pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table with all metadata fields
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding vector({self.embedding_dimension}),
                medical_domain VARCHAR(50),
                file_path TEXT,
                section TEXT,
                chunk_type VARCHAR(50),
                chunk_category VARCHAR(100),
                original_text TEXT,
                chunk_id VARCHAR(100),
                is_distilled BOOLEAN DEFAULT FALSE,
                guideline_title TEXT,
                source_file TEXT,
                evidence_certainty VARCHAR(20),
                recommendation_strength VARCHAR(20),
                
                -- Diabetes specific fields
                diabetes_type VARCHAR(20),
                technology_type VARCHAR(50),
                medication_type VARCHAR(50),
                outcome_type VARCHAR(50),
                
                -- Dermatitis specific fields
                condition_type VARCHAR(100),
                severity_classification VARCHAR(50),
                assessment_type VARCHAR(100),
                treatment_type VARCHAR(100),
                medication_class VARCHAR(50),
                scale_type VARCHAR(50),
                demographic_focus VARCHAR(50),
                
                -- Common fields
                population VARCHAR(50),
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Full metadata JSON for any additional fields
                metadata JSONB
            );
            """
            self.cursor.execute(create_table_query)
            
            # Create indexes for better search performance
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding ON {self.table_name} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_medical_domain ON {self.table_name} (medical_domain)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_is_distilled ON {self.table_name} (is_distilled)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_diabetes_type ON {self.table_name} (diabetes_type)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_population ON {self.table_name} (population)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_evidence_certainty ON {self.table_name} (evidence_certainty)",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_recommendation_strength ON {self.table_name} (recommendation_strength)"
            ]
            
            for index_query in indexes:
                try:
                    self.cursor.execute(index_query)
                except Exception as e:
                    logging.warning(f"Index creation warning: {str(e)}")
            
            self.conn.commit()
            logging.info(f"Database table '{self.table_name}' and indexes created successfully")
            
        except Exception as e:
            logging.error(f"Error setting up database: {str(e)}")
            self.conn.rollback()
            raise
    
    def add_documents(self, texts: List[str], embeddings_list: List[List[float]], metadatas: List[Dict[str, Any]]):
        """Add documents with embeddings and metadata to the database."""
        try:
            # Prepare data for bulk insert
            data = []
            for text, embedding, metadata in zip(texts, embeddings_list, metadatas):
                # Extract individual metadata fields
                row = (
                    text,  # content
                    embedding,  # embedding
                    metadata.get("medical_domain"),
                    metadata.get("file_path"),
                    metadata.get("section"),
                    metadata.get("chunk_type"),
                    metadata.get("chunk_category"),
                    metadata.get("original_text"),
                    metadata.get("chunk_id"),
                    metadata.get("is_distilled", False),
                    metadata.get("guideline_title"),
                    metadata.get("source_file"),
                    metadata.get("evidence_certainty"),
                    metadata.get("recommendation_strength"),
                    metadata.get("diabetes_type"),
                    metadata.get("technology_type"),
                    metadata.get("medication_type"),
                    metadata.get("outcome_type"),
                    metadata.get("condition_type"),
                    metadata.get("severity_classification"),
                    metadata.get("assessment_type"),
                    metadata.get("treatment_type"),
                    metadata.get("medication_class"),
                    metadata.get("scale_type"),
                    metadata.get("demographic_focus"),
                    metadata.get("population"),
                    json.dumps(metadata)  # Store full metadata as JSON
                )
                data.append(row)
            
            # Bulk insert using execute_values
            insert_query = f"""
            INSERT INTO {self.table_name} (
                content, embedding, medical_domain, file_path, section, chunk_type,
                chunk_category, original_text, chunk_id, is_distilled, guideline_title,
                source_file, evidence_certainty, recommendation_strength, diabetes_type,
                technology_type, medication_type, outcome_type, condition_type,
                severity_classification, assessment_type, treatment_type, medication_class,
                scale_type, demographic_focus, population, metadata
            ) VALUES %s
            """
            
            execute_values(self.cursor, insert_query, data)
            self.conn.commit()
            
            logging.info(f"Successfully inserted {len(data)} documents into database")
            
        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")
            self.conn.rollback()
            raise
    
    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logging.info("Database connection closed")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_from_json(json_data: Dict[str, Any]) -> str:
    """Extract relevant text from JSON structure for embedding - optimized for clinical queries."""
    texts = []
    
    # For distilled atomic facts, prioritize facts and concepts (our high-signal clinical content)
    if json_data.get("chunk_type") == "atomic_facts":
        # Add facts - these are our primary high-signal clinical content
        if "facts" in json_data:
            # Each fact is atomic and clinically optimized
            texts.extend(json_data["facts"])
        
        # Add concepts for comprehensive semantic matching (medical terminology, synonyms, abbreviations)
        if "concepts" in json_data:
            texts.extend(json_data["concepts"])
        
        # For distilled files, we focus only on the distilled content
        return " ".join(texts)
    
    # For non-distilled content (fallback for any remaining root files)
    # Add content or annotation if they exist
    for field in ["content", "annotation"]:
        if field in json_data:
            content = json_data[field]
            if isinstance(content, dict):
                # Flatten nested dictionary values
                texts.extend(str(v) for v in content.values() if v)
            elif isinstance(content, list):
                # Add list items
                texts.extend(str(item) for item in content if item)
            else:
                texts.append(str(content))
    
    # If no content found, try original_text as last resort
    if not texts and "original_text" in json_data.get("metadata", {}):
        texts.append(json_data["metadata"]["original_text"])
    
    return " ".join(texts)

def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    return embeddings.embed_query(text)

def prepare_metadata(json_data: Dict[str, Any], file_path: str, text: str, domain: str) -> Dict[str, Any]:
    """Prepare comprehensive metadata for clinical queries across multiple domains."""
    base_metadata = {
        "medical_domain": domain,
        "file_path": str(file_path),
        "section": json_data.get("metadata", {}).get("section", "") or json_data.get("section", ""),
        "chunk_type": json_data.get("chunk_type", ""),
        "chunk_category": json_data.get("chunk_category", ""),
        "original_text": json_data.get("metadata", {}).get("original_text", "") or json_data.get("original_text", ""),
        "chunk_id": json_data.get("chunk_id", ""),
        "is_distilled": json_data.get("chunk_type") == "atomic_facts",
        "guideline_title": json_data.get("metadata", {}).get("guideline_title", ""),
        "source_file": json_data.get("metadata", {}).get("source_file", ""),
        "evidence_certainty": "high" if "high certainty" in text.lower() else
                            "moderate" if "moderate certainty" in text.lower() else
                            "low" if "low certainty" in text.lower() else
                            "very_low" if "very low certainty" in text.lower() else "",
        "recommendation_strength": "strong" if "strong recommendation" in text.lower() or "green" in text.lower() else
                                 "conditional" if "conditional recommendation" in text.lower() or "yellow" in text.lower() else ""
    }
    
    # Domain-specific metadata
    if domain == "diabetes":
        diabetes_metadata = {
            "diabetes_type": "type_1" if "type 1" in text.lower() or "t1dm" in text.lower() else 
                           "type_2" if "type 2" in text.lower() or "t2dm" in text.lower() else "",
            "technology_type": "CGM" if "cgm" in text.lower() or "continuous glucose monitoring" in text.lower() else
                             "CSII" if "csii" in text.lower() or "insulin pump" in text.lower() else
                             "Flash" if "flash" in text.lower() else "",
            "medication_type": "SGLT2" if "sglt" in text.lower() else
                             "GLP1" if "glp" in text.lower() else
                             "DPP4" if "dpp" in text.lower() else
                             "metformin" if "metformin" in text.lower() else "",
            "outcome_type": "HbA1c" if "hba1c" in text.lower() else
                          "hypoglycemia" if "hypoglycaemia" in text.lower() or "hypoglycemia" in text.lower() else
                          "quality_of_life" if "quality of life" in text.lower() else "",
        }
        base_metadata.update(diabetes_metadata)
        
    elif domain == "dermatitis":
        # Extract dermatitis-specific metadata from clinical_metadata if available
        clinical_meta = json_data.get("clinical_metadata", {})
        dermatitis_metadata = {
            "condition_type": clinical_meta.get("condition_type", 
                            "atopic_dermatitis" if "atopic dermatitis" in text.lower() or "ad" in text.lower() else ""),
            "severity_classification": clinical_meta.get("severity_classification", ""),
            "assessment_type": clinical_meta.get("assessment_type", ""),
            "treatment_type": clinical_meta.get("treatment_type", ""),
            "medication_class": clinical_meta.get("medication_class", 
                              "topical" if "topical" in text.lower() else
                              "systemic" if "systemic" in text.lower() else
                              "biologic" if "dupilumab" in text.lower() or "jak" in text.lower() else ""),
            "scale_type": clinical_meta.get("scale_type", ""),
            "demographic_focus": clinical_meta.get("demographic_focus", ""),
        }
        base_metadata.update(dermatitis_metadata)

    # Common population metadata
    base_metadata["population"] = (
        json_data.get("clinical_metadata", {}).get("population", "") or
        ("children" if "children" in text.lower() or "adolescent" in text.lower() else
         "adults" if "adults" in text.lower() else "")
    )
    
    return base_metadata

def process_files():
    """Process all JSON files from multiple medical domains and store their embeddings in Neon pgvector."""
    
    # Initialize Neon pgvector store
    neon_connection_string = os.getenv("NEON_DATABASE_URL")
    if not neon_connection_string:
        raise ValueError("NEON_DATABASE_URL environment variable not set")
    
    vector_store = NeonPgVectorStore(
        connection_string=neon_connection_string,
        table_name="clinical_guidelines",
        embedding_dimension=1024  # mxbai-embed-large dimension
    )
    
    try:
        # Define medical domains to process
        medical_domains = ["diabetes", "dermatitis"]
        all_texts = []
        all_metadatas = []
        all_embeddings = []
        
        for domain in medical_domains:
            logging.info(f"Processing {domain.upper()} domain...")
            
            # PRIMARY SOURCE: Get all distilled files (our main optimized content)
            distilled_files = glob.glob(f"chunks/{domain}/distilled/*.json")
            logging.info(f"Found {len(distilled_files)} distilled {domain} JSON files to process (PRIMARY)")
            
            # SECONDARY SOURCE: Get remaining section files from root (if any not distilled)
            root_section_files = glob.glob(f"chunks/{domain}/section_[1-5]*.json")
            logging.info(f"Found {len(root_section_files)} root section {domain} JSON files to process (SECONDARY)")
            
            # Combine file paths - prioritize distilled files
            domain_files = distilled_files + root_section_files
            
            if not domain_files:
                logging.warning(f"No JSON files found in chunks/{domain} directory")
                continue
            
            logging.info(f"Total {domain} files to process: {len(domain_files)} (Distilled: {len(distilled_files)}, Root: {len(root_section_files)})")
            
            # Process domain files
            for file_path in tqdm(domain_files, desc=f"Processing {domain} files"):
                try:
                    logging.info(f"Processing file: {file_path}")
                    # Load and process JSON
                    json_data = load_json_file(file_path)
                    text = extract_text_from_json(json_data)
                    
                    # Skip if no meaningful text extracted
                    if not text.strip():
                        logging.warning(f"No text extracted from {file_path}, skipping")
                        continue
                    
                    # Get embedding
                    embedding = get_embedding(text)
                    
                    # Prepare comprehensive metadata for clinical queries
                    metadata = prepare_metadata(json_data, file_path, text, domain)
                    
                    all_texts.append(text)
                    all_metadatas.append(metadata)
                    all_embeddings.append(embedding)
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        # Add documents to Neon pgvector in batches
        if all_texts:
            logging.info(f"Adding {len(all_texts)} documents to Neon pgvector...")
            
            # Log distribution of content types
            distilled_count = sum(1 for m in all_metadatas if m.get("is_distilled"))
            logging.info(f"Content distribution: {distilled_count} distilled, {len(all_texts) - distilled_count} original")
            
            # Log medical domain distribution
            diabetes_count = sum(1 for m in all_metadatas if m.get("medical_domain") == "diabetes")
            dermatitis_count = sum(1 for m in all_metadatas if m.get("medical_domain") == "dermatitis")
            logging.info(f"Medical domain distribution: {diabetes_count} diabetes, {dermatitis_count} dermatitis")
            
            # Log diabetes-specific content distribution
            t1dm_count = sum(1 for m in all_metadatas if "type_1" in m.get("diabetes_type", ""))
            t2dm_count = sum(1 for m in all_metadatas if "type_2" in m.get("diabetes_type", ""))
            if diabetes_count > 0:
                logging.info(f"Diabetes type distribution: {t1dm_count} Type 1, {t2dm_count} Type 2")
            
            # Log dermatitis-specific content distribution
            ad_count = sum(1 for m in all_metadatas if "atopic_dermatitis" in m.get("condition_type", ""))
            if dermatitis_count > 0:
                logging.info(f"Dermatitis condition distribution: {ad_count} atopic dermatitis")
            
            # Process in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(all_texts), batch_size):
                batch_end = min(i + batch_size, len(all_texts))
                batch_texts = all_texts[i:batch_end]
                batch_embeddings = all_embeddings[i:batch_end]
                batch_metadatas = all_metadatas[i:batch_end]
                
                vector_store.add_documents(batch_texts, batch_embeddings, batch_metadatas)
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}")
            
            logging.info(f"Successfully processed and stored {len(all_texts)} documents in Neon pgvector")
            logging.info("Multi-domain clinical vector database ready for queries!")
        else:
            logging.warning("No texts to process!")
        
    except Exception as e:
        logging.error(f"Error in process_files: {str(e)}")
        raise
    finally:
        # Close database connection
        vector_store.close()

def verify_installation():
    """Verify that the embeddings were stored correctly."""
    neon_connection_string = os.getenv("NEON_DATABASE_URL")
    if not neon_connection_string:
        raise ValueError("NEON_DATABASE_URL environment variable not set")
    
    try:
        conn = psycopg2.connect(neon_connection_string)
        cursor = conn.cursor()
        
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM clinical_guidelines")
        total_count = cursor.fetchone()[0]
        logging.info(f"Total documents in database: {total_count}")
        
        # Check by medical domain
        cursor.execute("SELECT medical_domain, COUNT(*) FROM clinical_guidelines GROUP BY medical_domain")
        domain_counts = cursor.fetchall()
        for domain, count in domain_counts:
            logging.info(f"  {domain}: {count} documents")
        
        # Check by distilled vs original
        cursor.execute("SELECT is_distilled, COUNT(*) FROM clinical_guidelines GROUP BY is_distilled")
        distilled_counts = cursor.fetchall()
        for is_distilled, count in distilled_counts:
            logging.info(f"  {'Distilled' if is_distilled else 'Original'}: {count} documents")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logging.error(f"Error verifying installation: {str(e)}")

if __name__ == "__main__":
    try:
        logging.info("Starting embedding creation process for Neon pgvector...")
        
        # Load environment variables
        load_dotenv()
        
        # Check for required environment variable
        if not os.getenv("NEON_DATABASE_URL"):
            raise ValueError("Please set NEON_DATABASE_URL environment variable with your Neon connection string")
        
        # Process all files
        process_files()
        
        # Verify the installation
        logging.info("\nVerifying installation...")
        verify_installation()
        
        logging.info("\nEmbedding creation complete!")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        exit(1)