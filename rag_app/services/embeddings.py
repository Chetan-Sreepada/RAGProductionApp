from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        try:
            embeddings = self.model.encode(
                texts, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model
        
        Returns:
            Embedding dimension
        """
        return self.dimension