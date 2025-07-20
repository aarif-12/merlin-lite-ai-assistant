"""
Embedding model loader for document embeddings
Uses sentence-transformers/all-MiniLM-L6-v2 for free local embeddings
"""

from sentence_transformers import SentenceTransformer
import streamlit as st
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the embedding model"""
        self.model_name = model_name
        self.model = None
        
    @st.cache_resource
    def load_model(_self):
        """Load the embedding model with caching"""
        try:
            logger.info(f"Loading embedding model: {_self.model_name}")
            _self.model = SentenceTransformer(_self.model_name)
            logger.info("Embedding model loaded successfully")
            return _self.model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise e
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if self.model is None:
            self.model = self.load_model()
        
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise e
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.get_embeddings([text])[0]

# Global instance
embedding_model = EmbeddingModel()
