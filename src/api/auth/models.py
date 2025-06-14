from pydantic import BaseModel, EmailStr
from typing import Optional


class UserModel(BaseModel):
    """User model for database operations"""

    user_id: str
    user_email: EmailStr
    user_name: str
    user_pic: Optional[str] = None
    first_logged_in: Optional[int] = None
    last_accessed: Optional[int] = None


class TokenModel(BaseModel):
    """Token model for database operations"""

    access_token: str
    user_email: str
    session_id: str


class QueryModel(BaseModel):
    """Query model for database operations"""

    query_id: str
    user_id: str
    user_email: EmailStr
    answer: str


class FirebaseUserModel(BaseModel):
    """Firebase user model"""

    uid: str
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    email_verified: Optional[bool] = None
    provider_id: Optional[str] = None


class AuthTokenModel(BaseModel):
    """Authentication token model"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: FirebaseUserModel


class LoginRequest(BaseModel):
    """Login request model"""

    id_token: str


class LoginResponse(BaseModel):
    """Login response model"""

    message: str
    access_token: str
    user: FirebaseUserModel
