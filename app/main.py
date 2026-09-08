import json
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.engine import engine
from app.schemas import (
    FaceRegistrationResponse,
    FaceVerificationResponse,
    HealthCheckResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[-] Starting Axis Biometric Microservice...")
    print(
        f"[-] Loading {settings.INSIGHTFACE_MODEL_NAME} using {settings.EXECUTION_PROVIDER}..."
    )
    engine.initialize()
    print("[+] Model loaded and Biometric Engine is ready.")
    yield
    print("[-] Shutting down Biometric Microservice.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Axis Pharmaceuticals - Biometric Verification Engine
    
    Provides high-assurance facial embedding extraction and verification powered by ArcFace deep neural networks.
    
    * **Multi-Angle Registration:** Fuses Frontal, Left (30°), and Right (30°) facial profiles into a robust 512-dimension unit vector.
    * **1:1 Face Verification:** Compares live snapshot embeddings against stored Oracle/Postgres master vectors via Cosine Distance.
    * **Anti-Spoofing Rules:** Enforces single-face isolation and minimum detection confidence thresholds.
    """,
    openapi_tags=[
        {"name": "System", "description": "Liveness & Diagnostic endpoints"},
        {
            "name": "Biometrics",
            "description": "Face registration and verification procedures",
        },
    ],
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Health and Readiness Check",
    description="Validates server readiness and verifies that InsightFace models are loaded in memory.",
)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if engine.is_ready else "initializing",
        model_loaded=engine.is_ready,
        execution_provider=settings.EXECUTION_PROVIDER,
        version=settings.APP_VERSION,
    )


@app.post(
    "/api/v1/face/register",
    response_model=FaceRegistrationResponse,
    tags=["Biometrics"],
    status_code=status.HTTP_201_CREATED,
    summary="Register Face Profile (3-Angle Capture)",
    description="""
    Receives 3 distinct head poses captured from the Flutter camera stream:
    1. **center_frame:** Neutral frontal pose (Yaw ~ 0°)
    2. **left_frame:** Head rotated ~20° to 35° Left
    3. **right_frame:** Head rotated ~20° to 35° Right
    
    Averages vectors and returns a single 512-D unit array to be saved in the database.
    """,
)
async def register_face(
    cnic: str = Form(
        ...,
        description="User's unique national identifier",
        examples=["33100-1234567-1"],
    ),
    center_frame: UploadFile = File(
        ..., description="Front-facing image file (JPEG/PNG)"
    ),
    left_frame: UploadFile = File(
        ..., description="Left-turned profile image file (JPEG/PNG)"
    ),
    right_frame: UploadFile = File(
        ..., description="Right-turned profile image file (JPEG/PNG)"
    ),
):
    master_vector = engine.combine_embeddings(
        center_bytes=await center_frame.read(),
        left_bytes=await left_frame.read(),
        right_bytes=await right_frame.read(),
    )

    return FaceRegistrationResponse(
        success=True,
        message="Face profile successfully created from 3-pose capture.",
        cnic=cnic,
        embedding=master_vector.tolist(),
        dimension=len(master_vector),
    )


@app.post(
    "/api/v1/face/verify",
    response_model=FaceVerificationResponse,
    tags=["Biometrics"],
    summary="Verify 1:1 Live Face Against Stored Profile",
    description="""
    Performs one-to-one cosine matching between a live snapshot and the user's stored master vector.
    * Match threshold defaults to **0.68** (Bank standard, minimizing False Acceptance Rate).
    """,
)
async def verify_face(
    stored_embedding_json: str = Form(
        ...,
        description="JSON array string of exactly 512 floats fetched from the database",
        examples=["[-0.0412, 0.0251, 0.0883]"],
    ),
    live_frame: UploadFile = File(
        ...,
        description="Live snapshot taken during authentication",
    ),
):
    try:
        raw_list = json.loads(stored_embedding_json)
        stored_vector = np.array(raw_list, dtype=np.float32)
        if stored_vector.shape != (512,):
            raise ValueError()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="INVALID_EMBEDDING_PAYLOAD: Must be a valid JSON array of exactly 512 numbers.",
        )

    live_vector = engine.extract_embedding(await live_frame.read())
    score = engine.compute_similarity(live_vector, stored_vector)
    is_match = score >= settings.SIMILARITY_THRESHOLD

    if score >= 0.78:
        tier = "HIGH"
    elif score >= settings.SIMILARITY_THRESHOLD:
        tier = "MEDIUM"
    elif score >= 0.55:
        tier = "LOW"
    else:
        tier = "REJECTED"

    return FaceVerificationResponse(
        success=True,
        message="Face verified successfully."
        if is_match
        else "Verification failed. Score below threshold.",
        verified=is_match,
        similarity_score=round(score, 4),
        threshold=settings.SIMILARITY_THRESHOLD,
        confidence_tier=tier,
    )