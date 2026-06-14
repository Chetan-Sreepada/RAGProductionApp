from typing import List
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Text chunking utility with configurable chunk size and overlap
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info(f"TextChunker initialized: size={chunk_size}, overlap={overlap}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        # Split into sentences for better boundaries
        sentences = self._split_into_sentences(text)
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size and we have content, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Handle overlap: keep last few sentences for overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            # Add current sentence
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        # If no chunks were created (very short text), return the whole text as one chunk
        if not chunks and text:
            chunks = [text]
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace and cleaning
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for better chunking boundaries
        """
        # Simple sentence splitting (can be enhanced)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document texts
            
        Returns:
            List of chunk lists for each document
        """
        return [self.chunk_text(doc) for doc in documents]