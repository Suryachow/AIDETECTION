"""
API route handlers for the AI Text Humanizer.
This file contains additional API endpoints beyond the main humanize endpoint.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time

# Import local modules
from app.core.agent import HumanizerAgent
from app.core.rag import RAGSystem
from app.models import get_models

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define request and response models
class FeedbackRequest(BaseModel):
    original_text: str
    humanized_text: str
    user_rating: int  # 1-5 scale
    comments: Optional[str] = None

class BatchHumanizeRequest(BaseModel):
    texts: List[str]
    style: Optional[str] = "casual"
    preserve_meaning: Optional[bool] = True

class BatchHumanizeResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_time: float

# Get dependencies
def get_humanizer_agent():
    models = get_models()
    rag_system = RAGSystem()
    return HumanizerAgent(models=models, rag_system=rag_system)

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Submit feedback on humanized text to improve the system.
    
    Args:
        request: The feedback request
        background_tasks: FastAPI background tasks
        
    Returns:
        Acknowledgement of feedback receipt
    """
    try:
        # Process feedback in the background
        background_tasks.add_task(process_feedback, request)
        
        return {"status": "Feedback received", "message": "Thank you for your feedback"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-humanize", response_model=BatchHumanizeResponse)
async def batch_humanize(
    request: BatchHumanizeRequest,
    humanizer_agent: HumanizerAgent = Depends(get_humanizer_agent)
):
    """
    Humanize multiple texts in a single request.
    
    Args:
        request: The batch request containing texts to humanize
        humanizer_agent: The humanizer agent dependency
        
    Returns:
        Batch of humanized texts and metadata
    """
    try:
        start_time = time.time()
        
        results = []
        for text in request.texts:
            result = humanizer_agent.humanize(
                text=text,
                style=request.style,
                preserve_meaning=request.preserve_meaning,
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "total_time": total_time
        }
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def process_feedback(feedback: FeedbackRequest):
    """Process and store user feedback to improve the system."""
    # In a real implementation, this would:
    # 1. Store the feedback in a database
    # 2. Potentially use it for model fine-tuning
    # 3. Update reward models
    
    logger.info(f"Processing feedback with rating: {feedback.user_rating}")
    
    # Simulate processing time
    time.sleep(1)
    
    logger.info("Feedback processed and stored")
