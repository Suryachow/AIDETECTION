from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from services.ai_service import ai_service

router = APIRouter()

class HumanizeRequest(BaseModel):
    text: str
    passes: int = 2
    mode: str = "universal"
    intensity: Optional[int] = None

class DetectionRequest(BaseModel):
    text: str

@router.post("/humanize")
async def humanize_content(request: HumanizeRequest):
    try:
        return await ai_service.humanize(
            request.text, request.passes, request.mode, request.intensity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-ai")
async def detect_ai_content(request: DetectionRequest):
    try:
        result = await ai_service.detect_ai_content(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    return {
        "model_type": ai_service.model_type,
        "is_loaded": ai_service.detector_model is not None
    }
