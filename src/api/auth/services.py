import time
from datetime import UTC, datetime, timedelta
from typing import Annotated, Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Cookie, Depends, HTTPException, status
from firebase_admin import exceptions as firebase_exceptions
from firebase_admin import auth as firebase_auth
from jose import ExpiredSignatureError, JWTError, jwt
from loguru import logger
from firebase_admin.auth import verify_id_token
from api.config import env_vars
from api.db.tables import UserTable
from .models import QueryModel, TokenModel, UserModel

# JWT Configuration
ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 30

bearer_scheme = HTTPBearer(auto_error=False)


class FirebaseAuthService:
    """Firebase Authentication Service"""

    @staticmethod
    def verify_firebase_token(id_token: str) -> dict:
        """
        Verify Firebase ID token and return user info

        Args:
            id_token: Firebase ID token from client

        Returns:
            dict: User information from Firebase

        Raises:
            HTTPException: If token is invalid
        """
        try:
            # Verify the ID token
            decoded_token = firebase_auth.verify_id_token(id_token)
            return decoded_token
        except firebase_exceptions.InvalidArgumentError:
            logger.error("Invalid Firebase token format")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format")
        except firebase_exceptions.FirebaseError as e:
            logger.error(f"Firebase authentication error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")
        except Exception as e:
            logger.error(f"Unexpected error during token verification: {str(e)}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

    @staticmethod
    def create_custom_token(uid: str, additional_claims: Optional[dict] = None) -> str:
        """
        Create a custom Firebase token

        Args:
            uid: User ID
            additional_claims: Additional claims to include in token

        Returns:
            str: Custom Firebase token
        """
        try:
            return firebase_auth.create_custom_token(uid, additional_claims)
        except Exception as e:
            logger.error(f"Error creating custom token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create authentication token"
            )

    @staticmethod
    def get_user_by_uid(uid: str):
        """
        Get user information from Firebase by UID

        Args:
            uid: User ID

        Returns:
            UserRecord: Firebase user record
        """
        try:
            return firebase_auth.get_user(uid)
        except firebase_auth.UserNotFoundError:
            logger.error(f"User not found: {uid}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user information"
            )

    @staticmethod
    def get_user_by_email(email: str):
        """
        Get user information from Firebase by email

        Args:
            email: User email

        Returns:
            UserRecord: Firebase user record
        """
        try:
            return firebase_auth.get_user_by_email(email)
        except firebase_auth.UserNotFoundError:
            logger.error(f"User not found: {email}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get user information"
            )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time

    Returns:
        str: JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, env_vars.JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_jwt_token(token: str) -> dict:
    """
    Verify JWT token and return payload

    Args:
        token: JWT token

    Returns:
        dict: Token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = firebase_auth.verify_id_token(token)
        user_id: str = payload.get("sub")
        user_email: str = payload.get("email")

        if user_id is None or user_email is None:
            raise credentials_exception

        return payload

    except ExpiredSignatureError:
        logger.error("JWT token has expired")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please login again.")
    except JWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error during JWT verification: {str(e)}")
        raise credentials_exception


async def get_current_user(token: Optional[str] = Cookie(None, alias="access_token")) -> dict:
    """
    Get current user from JWT token in cookie

    Args:
        token: JWT token from cookie

    Returns:
        dict: User information

    Raises:
        HTTPException: If not authenticated
    """
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = verify_jwt_token(token)
    return {
        "user_id": payload.get("sub"),
        "user_email": payload.get("email"),
        "user_name": payload.get("name"),
        "user_pic": payload.get("picture"),
    }


async def get_optional_user(token: Optional[str] = Cookie(None, alias="access_token")) -> Optional[dict]:
    """
    Get current user if authenticated, otherwise return None

    Args:
        token: JWT token from cookie

    Returns:
        dict or None: User information if authenticated, None otherwise
    """
    if not token:
        return None

    try:
        return await get_current_user(token)
    except HTTPException:
        return None


def create_user(
    user_id: str,
    user_email: str,
    user_name: str,
    user_pic: Optional[str] = None,
    first_logged_in: Optional[int] = None,
    last_accessed: Optional[int] = None,
) -> UserTable:
    """
    Create or update user information in the database

    Args:
        user_id: User ID
        user_email: User email
        user_name: User name
        user_pic: User profile picture URL
        first_logged_in: First login timestamp
        last_accessed: Last accessed timestamp

    Returns:
        UserTable: Created user record
    """
    try:
        # Validate with Pydantic
        user = UserModel(
            user_id=user_id,
            user_email=user_email,
            user_name=user_name,
            user_pic=user_pic,
            first_logged_in=first_logged_in or int(time.time()),
            last_accessed=last_accessed or int(time.time()),
        )

        # Save to DynamoDB
        item = UserTable(
            pk=f"USER#{user.user_id}",
            sk="PROFILE",
            user_id=user.user_id,
            user_email=user.user_email,
            user_name=user.user_name,
            user_pic=str(user.user_pic) if user.user_pic else None,
            first_logged_in=user.first_logged_in,
            last_accessed=user.last_accessed,
        )
        item.save()
        logger.info(f"User created/updated: {user_email}")
        return item

    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")


def create_token(access_token: str, user_email: str, session_id: str) -> UserTable:
    """
    Store token information in the database

    Args:
        access_token: JWT access token
        user_email: User email
        session_id: Session ID

    Returns:
        UserTable: Created token record
    """
    try:
        # Validate with Pydantic
        token = TokenModel(access_token=access_token, user_email=user_email, session_id=session_id)

        # Save to DynamoDB
        item = UserTable(
            pk=f"TOKEN#{token.access_token}",
            sk=f"EMAIL#{token.user_email}",
            access_token=token.access_token,
            user_email=token.user_email,
            session_id=token.session_id,
        )
        item.save()
        logger.info(f"Token stored for user: {user_email}")
        return item

    except Exception as e:
        logger.error(f"Error storing token: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store token")


def create_query(query_id: str, user_id: str, user_email: str, answer: str) -> UserTable:
    """
    Store query information in the database

    Args:
        query_id: Query ID
        user_id: User ID
        user_email: User email
        answer: Query answer

    Returns:
        UserTable: Created query record
    """
    try:
        # Validate with Pydantic
        query = QueryModel(query_id=query_id, user_id=user_id, user_email=user_email, answer=answer)

        # Save to DynamoDB
        item = UserTable(
            pk=f"USER#{query.user_id}",
            sk=f"QUERY#{query.query_id}",
            query_id=query.query_id,
            user_id=query.user_id,
            user_email=query.user_email,
            answer=query.answer,
        )
        item.save()
        logger.info(f"Query stored for user: {user_email}")
        return item

    except Exception as e:
        logger.error(f"Error storing query: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store query")


# Dependency for protected routes
UserDependency = Annotated[dict, Depends(get_current_user)]
OptionalUserDependency = Annotated[Optional[dict], Depends(get_optional_user)]
