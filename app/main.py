import json
import httpx
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.engine import engine
from app.schemas import (
    FaceRegistrationResponse,
    FaceVerificationResponse,
    FaceLoginResponse,
    HealthCheckResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[-] Starting Axis Biometric Microservice...")
    engine.initialize()
    print("[+] Model loaded and Biometric Engine is ready.")
    yield
    print("[-] Shutting down Biometric Microservice.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Biometric Engine for Customer, Admin, and Guest Face Authentication",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
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
)
async def register_face(
    cnic: str = Form(...),
    center_frame: UploadFile = File(...),
    left_frame: UploadFile = File(...),
    right_frame: UploadFile = File(...),
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
)
async def verify_face(
    stored_embedding_json: str = Form(...), live_frame: UploadFile = File(...)
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


@app.post(
    "/api/v1/face/login",
    response_model=FaceLoginResponse,
    tags=["Biometrics"],
    summary="Multi-Role Face-Only 1:N Search & Session Initializer",
    description="""
    Performs 1:N face identification scoped to the requested role (CUSTOMER, ADMIN, or GUEST).
    Upon HIGH-tier match (>= 0.78), executes ORDS POST procedure to build the active session.
    """,
)
async def face_login(
    live_frame: UploadFile = File(...),
    role: str = Form("CUSTOMER"),
    device_id: str = Form("UNKNOWN"),
    auth_token: str = Form("UNKNOWN"),
    fcm_token: str = Form(""),
):
    normalized_role = role.strip().upper()

    # 1. Fetch registered faces from ORDS filtered by the selected role
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                settings.ORDS_FACE_LOGIN_URL,
                params={"p_role": normalized_role},
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to fetch {normalized_role} face database from ORDS.",
                )
            ords_data = response.json()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Error connecting to ORDS GET endpoint: {str(e)}",
            )

    faces_list = ords_data.get("faces", [])
    if not faces_list:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active registered face profiles found for role: {normalized_role}.",
        )

    # 2. Extract 512-D embedding from the incoming live camera snapshot
    live_vector = engine.extract_embedding(await live_frame.read())

    best_match_user = None
    best_score = -1.0

    # 3. Perform 1:N Cosine Distance search across candidates
    for record in faces_list:
        try:
            emb_json = record.get("EMBEDDING_JSON")
            if not emb_json:
                continue

            stored_list = json.loads(emb_json)
            stored_vector = np.array(stored_list, dtype=np.float32)

            if stored_vector.shape == (512,):
                score = engine.compute_similarity(live_vector, stored_vector)
                if score > best_score:
                    best_score = score
                    best_match_user = record
        except Exception:
            continue

    # 4. Enforce strict HIGH-confidence threshold (>= 0.78)
    HIGH_TIER_THRESHOLD = 0.78
    if best_match_user and best_score >= HIGH_TIER_THRESHOLD:
        matched_identifier = best_match_user.get(
            "IDENTIFIER"
        ) or best_match_user.get("CNIC_NO")
        matched_role = (
            best_match_user.get("ROLE") or normalized_role
        ).upper()

        # 5. Initialize active session in Oracle via ORDS POST
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                post_payload = {
                    "P_IDENTIFIER": matched_identifier,
                    "P_ROLE": matched_role,
                    "P_DEVICE_ID": device_id,
                    "P_AUTH_TOKEN": auth_token,
                    "P_FCM_TOKEN": fcm_token,
                }
                ords_post_res = await client.post(
                    settings.ORDS_FACE_LOGIN_URL, json=post_payload
                )
                session_data = ords_post_res.json()

                if session_data.get("status") == "success":
                    return FaceLoginResponse(
                        success=True,
                        message=f"{matched_role.capitalize()} face login successful.",
                        matched=True,
                        role=matched_role,
                        identifier=matched_identifier,
                        similarity_score=round(best_score, 4),
                        user_data=session_data,
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=session_data.get(
                            "message", "ORDS session creation failed."
                        ),
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error executing ORDS POST session creation: {str(e)}",
                )

    return FaceLoginResponse(
        success=False,
        message=f"No matching {normalized_role} face found or confidence below threshold.",
        matched=False,
        role=normalized_role,
        similarity_score=round(best_score, 4) if best_score != -1.0 else 0.0,
    )