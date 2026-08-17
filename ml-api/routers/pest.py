from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Dict
from inference.pest_engine import predict_pest

router = APIRouter(prefix="/predict", tags=["Pest Classification"])

class PestPredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_class_probabilities: Dict[str, float]

@router.post("/pest", response_model=PestPredictionResponse)
async def predict_pest_disease(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (leaf / crop photo),
    runs MobileNetV2 inference, and returns predicted pest/disease classification.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image. Please upload an image file (e.g. JPG, PNG)."
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")
            
        result = predict_pest(image_bytes)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {str(e)}"
        )
