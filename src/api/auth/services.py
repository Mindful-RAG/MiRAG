import os
import time
from datetime import UTC, datetime, timedelta
import traceback
from typing import Annotated

from authlib.integrations.starlette_client import OAuth
from fastapi import Cookie, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from starlette import status
from starlette.config import Config

from api.db.tables import UserTable
from api.config import env_vars

from .models import QueryModel, TokenModel, UserModel

ALGORITHM = "HS256"

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or None
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET") or None

if GOOGLE_CLIENT_ID is None or GOOGLE_CLIENT_SECRET is None:
    raise Exception("Missing env variables")

config_data = {"GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID, "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET}

starlette_config = Config(environ=config_data)

oauth = OAuth(starlette_config)

oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, env_vars.JWT_SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Cookie(None)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, env_vars.JWT_SECRET_KEY, algorithms=[ALGORITHM])

        user_id: str = payload.get("sub")
        user_email: str = payload.get("email")

        if user_id is None or user_email is None:
            raise credentials_exception

        return {"user_id": user_id, "user_email": user_email}

    except ExpiredSignatureError:
        # Specifically handle expired tokens
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please login again.")
    except JWTError:
        # Handle other JWT-related errors
        traceback.print_exc()
        raise credentials_exception
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=401, detail="Not Authenticated")


def create_refresh_token(data: dict, expires_delta: timedelta):
    return create_access_token(data, expires_delta)


def decode_token(token):
    return jwt.decode(token, os.getenv("SECRET_KEY"), algorithms=ALGORITHM)


def token_expired(token: Annotated[str, Depends(oauth_bearer)]):
    try:
        payload = decode_token(token)
        if not datetime.fromtimestamp(payload.get("exp"), UTC) > datetime.now(UTC):
            return True
        return False

    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user.")


user_dependency = Annotated[dict, Depends(get_current_user)]


def create_user(
    user_id: str,
    user_email: str,
    user_name: str,
    user_pic: str = None,
    first_logged_in: int = None,
    last_accessed: int = None,
) -> UserTable:
    """
    Log user information to the database
    """
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
    return item


def create_token(access_token: str, user_email: str, session_id: str) -> UserTable:
    """
    Log token information to the database
    """
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
    return item


def create_query(query_id: str, user_id: str, user_email: str, answer: str) -> UserTable:
    """
    Log query information to the database
    """
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
    return item
