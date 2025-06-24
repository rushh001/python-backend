# app_state.py
from typing import Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class AppState:
    """Singleton to manage application state and connections"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.cohere_client = None
            self.groq_client = None
            self.db_connection = None
            self._initialized = True
    
    def initialize(self):
        """Initialize all connections - call this after server starts"""
        try:
            logger.info("Initializing application resources...")
            
            # Initialize Cohere
            self._init_cohere()
            
            # Initialize Groq
            self._init_groq()
            
            # Initialize Database
            self._init_database()
            
            logger.info("All resources initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize resources: {e}")
            return False
    
    def _init_cohere(self):
        """Initialize Cohere client"""
        logger.info("Initializing Cohere client...")
        import cohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found")
        self.cohere_client = cohere.Client(api_key)
        logger.info("Cohere client initialized")
    
    def _init_groq(self):
        """Initialize Groq client"""
        logger.info("Initializing Groq client...")
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")
        self.groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialized")
    
    def _init_database(self):
        """Initialize database connection"""
        logger.info("Initializing database connection...")
        import psycopg2
        db_url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("Database URL not found")
        self.db_connection = psycopg2.connect(db_url)
        logger.info("Database connection established")
    
    def get_cohere_client(self):
        if not self.cohere_client:
            self._init_cohere()
        return self.cohere_client
    
    def get_groq_client(self):
        if not self.groq_client:
            self._init_groq()
        return self.groq_client
    
    def get_db_connection(self):
        if not self.db_connection:
            self._init_database()
        return self.db_connection

# Global instance
app_state = AppState()