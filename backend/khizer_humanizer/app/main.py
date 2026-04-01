"""
FastAPI application entry point for the AI Text Humanizer.
Ultra-advanced version specifically designed to bypass sophisticated AI detection systems.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Dict, Any, Optional
import os
import time
import random

# Import local modules
from app.core.agent import HumanizerAgent
from app.core.rag import RAGSystem
from app.models import load_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Text Humanizer",
    description="API for humanizing AI-generated text",
    version="0.3.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
models = load_models()
rag_system = RAGSystem()
humanizer_agent = HumanizerAgent(models=models, rag_system=rag_system)

# Define request and response models
class HumanizeRequest(BaseModel):
    text: str
    style: Optional[str] = "casual"  # casual, professional, creative
    preserve_meaning: Optional[bool] = False  # Default to false for more aggressive transformation
    temperature: Optional[float] = 0.9  # Default higher temperature for more aggressive transformation
    multi_pass: Optional[bool] = True  # Enable multi-pass processing by default

class HumanizeResponse(BaseModel):
    humanized_text: str
    humanness_score: float
    processing_time: float
    method: Optional[str] = None
    passes: Optional[int] = None

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "AI Text Humanizer API is running",
        "method": humanizer_agent.llm_type,
        "message": "Ultra-advanced version specifically designed to bypass sophisticated AI detection systems",
        "version": "0.3.0 - Undetectable.ai Bypass Edition"
    }

@app.post("/humanize", response_model=HumanizeResponse)
async def humanize_text(request: HumanizeRequest):
    """
    Humanize AI-generated text to make it sound more natural and bypass detection.
    
    Args:
        request: The request containing text to humanize and options
        
    Returns:
        Humanized text and metadata
    """
    try:
        logger.info(f"Received humanize request with style: {request.style}")
        
        start_time = time.time()
        
        # Apply multi-pass processing if enabled
        if request.multi_pass:
            # First pass with moderate transformation
            first_pass = humanizer_agent.humanize(
                text=request.text,
                style=request.style,
                preserve_meaning=True,  # First pass preserves meaning
                temperature=request.temperature * 0.8,  # Slightly lower temperature for first pass
            )
            
            # Second pass with more aggressive transformation
            second_pass = humanizer_agent.humanize(
                text=first_pass["humanized_text"],
                style=request.style,
                preserve_meaning=request.preserve_meaning,
                temperature=request.temperature,
            )
            
            # Determine if we need a third pass based on humanness score
            if second_pass["humanness_score"] < 0.85 and random.random() < 0.7:
                # Third pass with maximum transformation
                result = humanizer_agent.humanize(
                    text=second_pass["humanized_text"],
                    style=request.style,
                    preserve_meaning=False,  # Third pass can be more aggressive
                    temperature=min(1.0, request.temperature * 1.2),  # Increase temperature for third pass
                )
                passes = 3
            else:
                result = second_pass
                passes = 2
        else:
            # Single pass processing
            result = humanizer_agent.humanize(
                text=request.text,
                style=request.style,
                preserve_meaning=request.preserve_meaning,
                temperature=request.temperature,
            )
            passes = 1
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Add additional randomization to bypass pattern detection
        humanized_text = result["humanized_text"]
        
        # Add random spacing variations (undetectable.ai specific bypass)
        if random.random() < 0.3:
            humanized_text = _add_random_spacing_variations(humanized_text)
        
        # Apply structural rhythm breaking (Originality.ai explicit bypass)
        humanized_text = _vary_sentence_length(humanized_text, intensity=0.7)
        
        # Add random unicode characters that look like standard punctuation
        if random.random() < 0.25:
            humanized_text = _add_unicode_variations(humanized_text)
        
        # Return the result with total processing time
        return {
            "humanized_text": humanized_text,
            "humanness_score": result["humanness_score"],
            "processing_time": total_processing_time,
            "method": f"{humanizer_agent.llm_type}-multipass" if request.multi_pass else humanizer_agent.llm_type,
            "passes": passes
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

import re

def _vary_sentence_length(text: str, intensity: float = 0.5) -> str:
    """Implement deep structural rhythm variation to bypass Originality.ai
    by dynamically breaking long sentences and restructuring rhythm."""
    sentences = re.split(r'([.!?]\s+)', text)
    result = []
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if not sentence:
            continue
            
        delimiter = sentences[i+1] if i+1 < len(sentences) else ""
        
        # Break long sentences
        if len(sentence) > 120 and random.random() < intensity:
            parts = re.split(r'(?:,\s+|\s+and\s+|\s+but\s+)', sentence, maxsplit=1)
            if len(parts) > 1:
                first_part = parts[0]
                second_part = parts[1].strip()
                if second_part:
                    second_part = second_part[0].upper() + second_part[1:]
                result.append(f"{first_part}.")
                result.append(f" {second_part}{delimiter}")
                continue
                
        result.append(f"{sentence}{delimiter}")
        
    return "".join(result)

def _add_random_spacing_variations(text: str) -> str:
    """Add random spacing variations to bypass pattern detection."""
    # Replace some spaces with non-breaking spaces or zero-width spaces
    special_spaces = ['\u00A0', '\u200B', '\u2009', '\u202F']
    
    words = text.split()
    result = []
    
    for word in words:
        result.append(word)
        # 10% chance to add a special space instead of a regular space
        if random.random() < 0.1:
            result.append(random.choice(special_spaces))
        else:
            result.append(' ')
    
    return ''.join(result[:-1])  # Remove the last space

def _add_unicode_variations(text: str) -> str:
    """Add unicode variations of standard punctuation."""
    # Map of standard punctuation to similar unicode characters
    unicode_variations = {
        '.': ['.', '．', '｡'],  # Period variations
        ',': [',', '，', '､'],  # Comma variations
        '!': ['!', '！', '﹗'],  # Exclamation variations
        '?': ['?', '？', '﹖'],  # Question mark variations
        '-': ['-', '－', '‐', '‑'],  # Hyphen variations
        '"': ['"', '＂', '"', '"'],  # Quotation mark variations
        "'": ["'", '＇', ''', ''']   # Apostrophe variations
    }
    
    result = ""
    for char in text:
        if char in unicode_variations and random.random() < 0.15:
            result += random.choice(unicode_variations[char])
        else:
            result += char
    
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": models.are_loaded(),
        "rag_system": rag_system.status(),
        "method": humanizer_agent.llm_type,
        "version": "0.3.0 - Undetectable.ai Bypass Edition"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
