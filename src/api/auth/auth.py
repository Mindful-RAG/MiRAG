import os
import traceback
import uuid
from datetime import datetime, timedelta

import requests
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import APIRouter, Cookie, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from jose import ExpiredSignatureError, JWTError, jwt
from loguru import logger
from starlette.config import Config

# from api.auth.services import create_token, create_user
from api.config import env_vars

load_dotenv()

router = APIRouter()

# config = Config(".env")

# # Setup OAuth2
# oauth = OAuth()

# oauth.register(
#     name="google",
#     client_id=config("GOOGLE_CLIENT_ID"),
#     client_secret=config("GOOGLE_CLIENT_SECRET"),
#     authorize_url="https://accounts.google.com/o/oauth2/auth",
#     authorize_params=None,
#     access_token_url="https://accounts.google.com/o/oauth2/token",
#     access_token_params=None,
#     refresh_token_url=None,
#     authorize_state=config("SECRET_KEY"),
#     redirect_uri="http://localhost:8000/auth",
#     jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
#     client_kwargs={"scope": "openid profile email"},
# )
# # JWT Configurations
# ALGORITHM = "HS256"
# JWT_EXPIRATION = 30  # minutes


# def create_access_token(data: dict, expires_delta: timedelta = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRATION))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, env_vars.JWT_SECRET_KEY, algorithm=ALGORITHM)


# def get_current_user(token: str = Cookie(None)):
#     if not token:
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     credentials_exception = HTTPException(
#         status_code=401,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, env_vars.JWT_SECRET_KEY, algorithms=[ALGORITHM])

#         user_id: str = payload.get("sub")
#         user_email: str = payload.get("email")

#         if user_id is None or user_email is None:
#             raise credentials_exception

#         return {"user_id": user_id, "user_email": user_email}

#     except ExpiredSignatureError:
#         # Specifically handle expired tokens
#         traceback.print_exc()
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please login again.")
#     except JWTError:
#         # Handle other JWT-related errors
#         traceback.print_exc()
#         raise credentials_exception
#     except Exception:
#         traceback.print_exc()
#         raise HTTPException(status_code=401, detail="Not Authenticated")


# @router.get("/login")
# async def login(request: Request):
#     request.session.clear()
#     referer = request.headers.get("referer")
#     frontend_url = os.getenv("FRONTEND_URL")
#     redirect_url = os.getenv("REDIRECT_URL")
#     request.session["login_redirect"] = frontend_url

#     return await oauth.google.authorize_redirect(request, redirect_url, prompt="consent")


# @router.route("/auth")
# async def auth(request: Request):
#     state_in_request = request.query_params.get("state")

#     logger.info(f"Request Session: {request.session}")

#     logger.info(f"Request state (from query params): {state_in_request}")

#     try:
#         token = await oauth.google.authorize_access_token(request)
#     except Exception as e:
#         logger.info(str(e))
#         raise HTTPException(status_code=401, detail="Google authentication failed.")

#     try:
#         user_info_endpoint = "https://www.googleapis.com/oauth2/v2/userinfo"
#         headers = {"Authorization": f"Bearer {token['access_token']}"}
#         google_response = requests.get(user_info_endpoint, headers=headers)
#         user_info = google_response.json()
#     except Exception as e:
#         logger.info(str(e))
#         raise HTTPException(status_code=401, detail="Google authentication failed.")

#     user = token.get("userinfo")
#     expires_in = token.get("expires_in")
#     user_id = user.get("sub")
#     iss = user.get("iss")
#     user_email = user.get("email")
#     first_logged_in = datetime.utcnow().toordinal()
#     last_accessed = datetime.utcnow().toordinal()

#     user_name = user_info.get("name")
#     user_pic = user_info.get("picture")

#     logger.info(f"User name:{user_name}")
#     logger.info(f"User Email:{user_email}")

#     if iss not in ["https://accounts.google.com", "accounts.google.com"]:
#         raise HTTPException(status_code=401, detail="Google authentication failed.")

#     if user_id is None:
#         raise HTTPException(status_code=401, detail="Google authentication failed.")

#     # Create JWT token
#     access_token_expires = timedelta(seconds=expires_in)
#     access_token = create_access_token(data={"sub": user_id, "email": user_email}, expires_delta=access_token_expires)

#     session_id = str(uuid.uuid4())
#     create_user(user_id, user_email, user_name, user_pic, first_logged_in, last_accessed)
#     create_token(access_token, user_email, session_id)

#     redirect_url = request.session.pop("login_redirect", "")
#     logger.info(f"Redirecting to: {redirect_url}")
#     response = RedirectResponse(redirect_url)
#     print(f"Access Token: {access_token}")
#     response.set_cookie(
#         key="token",
#         value=access_token,
#         httponly=True,
#         secure=True,  # Ensure you're using HTTPS
#         samesite="strict",  # Set the SameSite attribute to None
#     )

#     return response


# @router.get("/logout")
# async def logout(request: Request):
#     request.session.clear()
#     frontend_url = os.getenv("FRONTEND_URL", "/")
#     response = RedirectResponse(url=frontend_url, status_code=204)
#     response.delete_cookie("token")
#     return response
