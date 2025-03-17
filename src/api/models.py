from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
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
