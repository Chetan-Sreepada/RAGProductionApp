import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
import logging
from rag_app.services.embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS-based vector store with persistence
    """
    
    def __init__(self, embedding_service: EmbeddingService, persist_dir: str = "faiss_index"):
        """
        Initialize the vector store
        
        Args:
            embedding_service: Service for generating embeddings
            persist_dir: Directory to save FAISS index and metadata
        """
        self.embedding_service = embedding_service
        self.persist_dir = persist_dir
        self.dimension = embedding_service.get_embedding_dimension()
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents = []  # Store document texts
        self.metadata = []   # Store document metadata (doc_id, chunk_id)
        self.is_loaded = False
        
        # Try to load existing index
        self._load_index()
    
    def _get_index_path(self) -> str:
        """Get path for FAISS index file"""
        return os.path.join(self.persist_dir, "faiss_index.bin")
    
    def _get_metadata_path(self) -> str:
        """Get path for metadata file"""
        return os.path.join(self.persist_dir, "metadata.pkl")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self._get_index_path())
            
            # Save metadata and documents
            metadata = {
                'documents': self.documents,
                'metadata': self.metadata
            }
            with open(self._get_metadata_path(), 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved successfully with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata['documents']
                    self.metadata = metadata['metadata']
                
                self.is_loaded = True
                logger.info(f"Index loaded successfully with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self.is_loaded = False
        else:
            logger.info("No existing index found. Starting with empty index.")
            self.is_loaded = False
    
    def add_documents(self, chunks: List[str], doc_id: str):
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of text chunks
            doc_id: Document identifier
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        try:
            # Generate embeddings for all chunks
            embeddings = self.embedding_service.embed_texts(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and metadata
            start_idx = len(self.documents)
            for i, chunk in enumerate(chunks):
                self.documents.append(chunk)
                self.metadata.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_chunk_{start_idx + i}",
                    'index': start_idx + i
                })
            
            # Save to disk
            self._save_index()
            logger.info(f"Added {len(chunks)} chunks from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if self.index.ntotal == 0:
            logger.warning("No documents in index")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            k = min(k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(scores[0][i])
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_total_documents(self) -> int:
        """Get total number of documents in the store"""
        return self.index.ntotal
    
    def clear(self):
        """Clear all documents from the store"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadata = []
        self._save_index()
        logger.info("Vector store cleared")