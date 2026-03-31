"""
Face Recognition Microservice
FastAPI service for face enrollment and verification.
"""

import base64
import io
import os
import json
import logging
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from face_recognition_service import FaceRecognitionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="Microservice for face enrollment and verification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face recognition service
face_service = FaceRecognitionService()
DEFAULT_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))


# Request/Response Models
class EnrollRequest(BaseModel):
    employee_id: int
    image: str  # base64 encoded


class EnrollResponse(BaseModel):
    success: bool
    face_embedding: Optional[list] = None
    message: str


class VerifyRequest(BaseModel):
    image: str  # base64 encoded
    threshold: Optional[float] = 0.6


class VerifyResponse(BaseModel):
    success: bool
    matched: bool
    employee_id: Optional[int] = None
    confidence: Optional[float] = None
    message: str


class VerifySpecificRequest(BaseModel):
    employee_id: int
    image: str  # base64 encoded
    threshold: Optional[float] = 0.6


class HealthResponse(BaseModel):
    status: str
    service: str


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="face-recognition")


@app.post("/enroll", response_model=EnrollResponse)
async def enroll(request: EnrollRequest):
    """
    Enroll a new face for an employee.
    Extracts face embedding from image and stores it.
    """
    try:
        logger.info(f"Enrolling face for employee {request.employee_id}")
        
        # Decode image
        image_array = decode_base64_image(request.image)
        
        # Get face embedding
        embedding = face_service.get_face_encoding(image_array)
        
        if embedding is None:
            return EnrollResponse(
                success=False,
                message="No face detected in image. Please ensure your face is clearly visible."
            )
        
        # Store embedding
        face_service.store_embedding(request.employee_id, embedding)
        
        logger.info(f"Successfully enrolled face for employee {request.employee_id}")
        
        return EnrollResponse(
            success=True,
            face_embedding=embedding.tolist(),
            message="Face enrolled successfully"
        )
        
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    """
    Verify a face against all enrolled employees.
    Returns the best match if confidence exceeds threshold.
    """
    try:
        logger.info("Verifying face against enrolled employees")
        
        # Decode image
        image_array = decode_base64_image(request.image)
        
        # Get face embedding from captured image
        captured_embedding = face_service.get_face_encoding(image_array)
        
        if captured_embedding is None:
            return VerifyResponse(
                success=True,
                matched=False,
                message="No face detected in image. Please ensure your face is clearly visible."
            )
        
        # Compare against all enrolled faces
        results = face_service.compare_against_all(
            captured_embedding, 
            threshold=request.threshold
        )
        
        if results:
            # Get best match (highest confidence)
            best_match = max(results, key=lambda x: x['confidence'])
            
            logger.info(f"Face matched with employee {best_match['employee_id']} (confidence: {best_match['confidence']:.3f})")
            
            return VerifyResponse(
                success=True,
                matched=True,
                employee_id=best_match['employee_id'],
                confidence=best_match['confidence'],
                message="Face matched successfully"
            )
        else:
            logger.info("No matching face found")
            return VerifyResponse(
                success=True,
                matched=False,
                message="Face not recognized. Please try again or contact HR."
            )
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/verify-specific", response_model=VerifyResponse)
async def verify_specific(request: VerifySpecificRequest):
    """
    Verify a face against a specific employee.
    Used for re-verification scenarios.
    """
    try:
        logger.info(f"Verifying face against specific employee {request.employee_id}")
        
        # Decode image
        image_array = decode_base64_image(request.image)
        
        # Get face embedding from captured image
        captured_embedding = face_service.get_face_encoding(image_array)
        
        if captured_embedding is None:
            return VerifyResponse(
                success=True,
                matched=False,
                message="No face detected in image. Please ensure your face is clearly visible."
            )
        
        # Compare against specific employee
        match_result = face_service.compare_with_employee(
            request.employee_id,
            captured_embedding,
            threshold=request.threshold
        )
        
        if match_result:
            logger.info(f"Face verified for employee {request.employee_id} (confidence: {match_result['confidence']:.3f})")
            
            return VerifyResponse(
                success=True,
                matched=True,
                employee_id=request.employee_id,
                confidence=match_result['confidence'],
                message="Face verified successfully"
            )
        else:
            logger.info(f"Face verification failed for employee {request.employee_id}")
            return VerifyResponse(
                success=True,
                matched=False,
                employee_id=request.employee_id,
                message="Face does not match the specified employee"
            )
        
    except Exception as e:
        logger.error(f"Specific verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.get("/enrolled-employees")
async def get_enrolled_employees():
    """Get list of all enrolled employee IDs."""
    try:
        employees = face_service.get_all_enrolled_employees()
        return {"success": True, "employee_ids": employees}
    except Exception as e:
        logger.error(f"Failed to get enrolled employees: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/enroll/{employee_id}")
async def delete_enrollment(employee_id: int):
    """Remove face enrollment for an employee."""
    try:
        face_service.remove_embedding(employee_id)
        logger.info(f"Removed enrollment for employee {employee_id}")
        return {"success": True, "message": f"Enrollment removed for employee {employee_id}"}
    except Exception as e:
        logger.error(f"Failed to remove enrollment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
