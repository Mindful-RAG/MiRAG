from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class BaseChat(BaseModel):
    """
    Base class for chat messages.
    """

    timestamp: datetime = datetime.now()


class ChatResponse(BaseChat):
    """
    Response message in a chat.
    """

    role: Literal["assistant", "user"]
    content: str


class ChatRequest(BaseChat):
    """
    Request message in a chat.
    """

    role: Literal["user"]
    content: str
