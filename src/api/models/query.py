from pydantic import BaseModel


class QueryIn(BaseModel):
    query: str
    session_id: str


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
