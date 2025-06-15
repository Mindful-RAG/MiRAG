from pydantic import BaseModel
from typing import Optional


class QueryIn(BaseModel):
    query: str
    session_id: str
    custom_corpus_id: Optional[str] = None


class LongragOut(BaseModel):
    query: str
    short_answer: str
    long_answer: str
    markdown: str


class MiragOut(BaseModel):
    query: str
    short_answer: str
    long_answer: str
    status: str
    markdown: str


class UserCreate(BaseModel):
    email: str
    name: str
    picture: str = None


class Token(BaseModel):
    access_token: str
    token_type: str


class UploadResponse(BaseModel):
    file: str
    content_type: str
    size: int
    corpus_id: str
