from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from rag_app.services.embeddings import EmbeddingService
from rag_app.services.llm import LLMService
from rag_app.vectorstore.store import VectorStore
from rag_app.utils.chunking import TextChunker

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_service)
chunker = TextChunker(chunk_size=500, overlap=50)
llm_service = LLMService()

# Request/Response Models
class IngestRequest(BaseModel):
    text: str
    document_id: str = None

class IngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_stored: int
    message: str

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]
    status: str

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store.is_loaded,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm": "gemini-pro"
    }

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the RAG system
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate document ID if not provided
        doc_id = request.document_id or f"doc_{hash(request.text) % 1000000}"
        
        # Chunk the text
        chunks = chunker.chunk_text(request.text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from text")
        
        # Add to vector store
        vector_store.add_documents(chunks, doc_id)
        
        return IngestResponse(
            status="success",
            document_id=doc_id,
            chunks_stored=len(chunks),
            message=f"Successfully ingested {len(chunks)} chunks"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the RAG system
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if vector store has documents
        if vector_store.get_total_documents() == 0:
            return ChatResponse(
                answer="No documents have been ingested yet. Please add some documents first.",
                retrieved_contexts=[],
                status="no_documents"
            )
        
        # Retrieve relevant chunks
        retrieved_chunks = vector_store.similarity_search(request.query, k=request.top_k)
        
        if not retrieved_chunks:
            return ChatResponse(
                answer="No relevant context found for your query.",
                retrieved_contexts=[],
                status="no_context"
            )
        
        # Generate answer using LLM
        contexts = [chunk['text'] for chunk in retrieved_chunks]
        answer = llm_service.generate_answer(request.query, contexts)
        
        return ChatResponse(
            answer=answer,
            retrieved_contexts=contexts,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")