from typing import Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from .services import get_optional_user


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware to add user context to requests
    """

    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/auth/logout",
            "/auth/status",
            "/auth/login/google",
            "/auth/callback/google",
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Get user if authenticated
        try:
            user = await get_optional_user(request.cookies.get("access_token"))
            request.state.user = user
        except Exception:
            request.state.user = None

        response = await call_next(request)
        return response


class RequireAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to require authentication for all routes except excluded ones
    """

    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/auth/logout",
            "/auth/status",
            "/auth/login/google",
            "/auth/callback/google",
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)

        # Check if user is authenticated
        try:
            token = request.cookies.get("access_token")
            if not token:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

            user = await get_optional_user(token)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired authentication"
                )

            request.state.user = user

        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

        response = await call_next(request)
        return response
