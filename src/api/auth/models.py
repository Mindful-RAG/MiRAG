from pynamodb.attributes import UnicodeAttribute
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional
from pynamodb.models import Model
from sqlalchemy import Column, Integer, String


class UserModel(BaseModel):
    user_id: str
    user_email: EmailStr
    user_name: str
    user_pic: Optional[str] = None
    first_logged_in: Optional[int] = None
    last_accessed: Optional[int] = None


class TokenModel(BaseModel):
    access_token: str
    user_email: str
    session_id: str


class QueryModel(BaseModel):
    query_id: str
    user_id: str
    user_email: EmailStr
    answer: str
