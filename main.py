from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_app.api.routes import router
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app - MUST be named "app" for uvicorn
app = FastAPI(
    title="RAG Production API",
    description="Local RAG System with FAISS and Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["rag"])

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RAG Production API...")
    logger.info("Backend is ready to accept connections")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Production API...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Production API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "ingest": "POST /api/v1/ingest",
            "chat": "POST /api/v1/chat"
        }
    }

# This allows running with python main.py directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)