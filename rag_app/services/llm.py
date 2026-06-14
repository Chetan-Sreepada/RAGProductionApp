from groq import Groq
import os
from dotenv import load_dotenv
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        self.client = Groq(api_key=self.api_key)
        # Using the current working Llama 3.3 70B model
        self.model = "llama-3.3-70b-versatile"
        logger.info(f"Groq initialized with {self.model}")
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not query:
            return "Please provide a question."
        
        if not contexts:
            return "No documents found. Please upload a document first."
        
        context_text = "\n\n---\n\n".join(contexts)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "Answer based ONLY on the provided context. Say if info is not found."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content.strip()