import uuid
from datetime import UTC, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, EmailStr

from .services import (
    FirebaseAuthService,
    create_access_token,
    create_token,
    create_user,
    get_current_user,
    UserDependency,
)

router = APIRouter()


class FirebaseTokenRequest(BaseModel):
    """Request model for Firebase token authentication"""

    id_token: str


class AuthResponse(BaseModel):
    """Response model for authentication"""

    message: str
    user: dict
    access_token: Optional[str] = None


class UserResponse(BaseModel):
    """Response model for user information"""

    user_id: str
    user_email: EmailStr
    user_name: str
    user_pic: Optional[str] = None


@router.post("/login", response_model=AuthResponse)
async def login_with_firebase(token_request: FirebaseTokenRequest):
    """
    Authenticate user with Firebase ID token

    This endpoint accepts a Firebase ID token from the client (after Google OAuth),
    verifies it, creates a session, and returns a JWT token.
    """
    try:
        # Verify Firebase ID token
        firebase_user = FirebaseAuthService.verify_firebase_token(token_request.id_token)

        user_id = firebase_user.get("uid")
        user_email = firebase_user.get("email")
        user_name = firebase_user.get("name", "")
        user_pic = firebase_user.get("picture")

        if not user_id or not user_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user information from Firebase"
            )

        # Create or update user in database
        current_time = int(datetime.now(UTC).timestamp())
        create_user(
            user_id=user_id,
            user_email=user_email,
            user_name=user_name,
            user_pic=user_pic,
            first_logged_in=current_time,
            last_accessed=current_time,
        )

        # Create JWT access token
        token_data = {
            "sub": user_id,
            "email": user_email,
            "name": user_name,
            "picture": user_pic,
        }
        access_token = create_access_token(
            data=token_data,
            expires_delta=timedelta(hours=24),  # 24 hour token
        )

        # Store token in database
        session_id = str(uuid.uuid4())
        create_token(access_token, user_email, session_id)

        logger.info(f"User authenticated successfully: {user_email}")

        return AuthResponse(
            message="Authentication successful",
            user={
                "user_id": user_id,
                "user_email": user_email,
                "user_name": user_name,
                "user_pic": user_pic,
            },
            access_token=access_token,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Authentication failed")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserDependency):
    """
    Get current authenticated user information
    """
    return UserResponse(
        user_id=current_user["user_id"],
        user_email=current_user["user_email"],
        user_name=current_user["user_name"],
        user_pic=current_user.get("user_pic"),
    )


@router.post("/logout")
async def logout():
    """
    Logout user by clearing session

    Note: This endpoint clears the client-side token.
    The actual token invalidation should be handled on the client side.
    """
    response = JSONResponse(content={"message": "Logged out successfully"}, status_code=status.HTTP_200_OK)
    response.delete_cookie(key="access_token", httponly=True, secure=True, samesite="strict")
    return response


@router.get("/status")
async def auth_status(request: Request):
    """
    Check authentication status
    """
    try:
        # Try to get user from token
        token = request.cookies.get("access_token")
        if not token:
            return JSONResponse(
                content={"authenticated": False, "message": "No token found"}, status_code=status.HTTP_200_OK
            )

        current_user = await get_current_user(token)
        return JSONResponse(
            content={"authenticated": True, "user": current_user, "message": "User is authenticated"},
            status_code=status.HTTP_200_OK,
        )

    except HTTPException:
        return JSONResponse(
            content={"authenticated": False, "message": "Invalid or expired token"}, status_code=status.HTTP_200_OK
        )


# OAuth redirect endpoints for frontend integration
@router.get("/login/google")
async def google_login_redirect():
    """
    Redirect to Google OAuth

    This endpoint should be called from the frontend to initiate Google OAuth.
    The actual OAuth flow should be handled by Firebase on the client side.
    """
    return JSONResponse(
        content={
            "message": "Use Firebase SDK on the client side for Google authentication",
            "auth_url": "https://firebase.google.com/docs/auth/web/google-signin",
        },
        status_code=status.HTTP_200_OK,
    )


@router.get("/callback/google")
async def google_callback():
    """
    Google OAuth callback

    This endpoint is for reference. The actual callback should be handled
    by Firebase on the client side, then the ID token should be sent to /login.
    """
    return JSONResponse(
        content={
            "message": "OAuth callback should be handled by Firebase SDK on client side",
            "next_step": "Send the Firebase ID token to /auth/login endpoint",
        },
        status_code=status.HTTP_200_OK,
    )


@router.get("/userid")
async def get_user_id(user: UserDependency):
    return {"id": user["user_id"]}
