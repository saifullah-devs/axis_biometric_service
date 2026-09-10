from typing import List, Optional
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    success: bool
    message: str


class FaceRegistrationResponse(BaseResponse):
    cnic: str = Field(
        ...,
        description="CNIC / Identifier of the registered user",
        examples=["33100-1234567-1"],
    )
    embedding: List[float] = Field(
        ...,
        description="Averaged 512-D unit vector representation",
        min_length=512,
        max_length=512,
    )
    dimension: int = Field(default=512, examples=[512])


class FaceVerificationResponse(BaseResponse):
    verified: bool = Field(
        ...,
        description="True if similarity score meets bank-grade threshold",
    )
    similarity_score: float = Field(
        ...,
        description="Cosine similarity between -1.0 and 1.0",
        examples=[0.8241],
    )
    threshold: float = Field(
        ...,
        description="Required pass threshold",
        examples=[0.68],
    )
    confidence_tier: str = Field(
        ...,
        description="HIGH, MEDIUM, LOW, or REJECTED",
        examples=["HIGH"],
    )


class FaceLoginResponse(BaseResponse):
    matched: bool = Field(
        ..., description="True if a matching face met the HIGH tier threshold"
    )
    role: Optional[str] = Field(
        None, description="Matched role: CUSTOMER, ADMIN, or GUEST"
    )
    identifier: Optional[str] = Field(
        None, description="Matched CNIC (Customer) or Username (Admin/Guest)"
    )
    similarity_score: Optional[float] = Field(
        None, description="Achieved cosine similarity score"
    )
    user_data: Optional[dict] = Field(
        None, description="Session payload returned from ORDS on success"
    )


class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    execution_provider: str
    version: str