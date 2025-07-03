import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
import boto3
from boto3.dynamodb.conditions import Key
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from llama_index.storage.chat_store.dynamodb.base import DynamoDBChatStore

from api.config import env_vars
from api.models.query import ChatSession, ChatSessionCreate, ChatSessionUpdate, ChatSessionList
from api.utils.observability import logger


class ChatStoreService:
    """Service for managing chat sessions and persistence with DynamoDB"""

    def __init__(self):
        self.chat_store = DynamoDBChatStore(
            table_name=env_vars.CHAT_STORE_TABLE_NAME,
            # aws_access_key_id=env_vars.AWS_ACCESS_KEY_ID,
            # aws_secret_access_key=env_vars.AWS_SECRET_ACCESS_KEY,
            region_name=env_vars.AWS_REGION,
        )

        # Separate DynamoDB client for session metadata
        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=env_vars.AWS_REGION,
            # aws_access_key_id=env_vars.AWS_ACCESS_KEY_ID,
            # aws_secret_access_key=env_vars.AWS_SECRET_ACCESS_KEY,
        )

        # Table for session metadata (separate from chat messages)
        self.sessions_table = self.dynamodb.Table(f"{env_vars.CHAT_STORE_TABLE_NAME}-sessions")

        logger.info("ChatStoreService initialized with DynamoDB")

    def get_memory_for_user_session(
        self, user_email: str, session_id: str, session_type: str = "mirag"
    ) -> ChatMemoryBuffer:
        """Get or create a chat memory buffer for a specific user session"""
        chat_store_key = f"{user_email}:{session_id}:{session_type}"

        return ChatMemoryBuffer.from_defaults(
            token_limit=env_vars.CHAT_TOKEN_LIMIT,
            chat_store=self.chat_store,
            chat_store_key=chat_store_key,
        )

    def get_memory_for_anonymous_session(self, session_id: str, session_type: str = "mirag") -> ChatMemoryBuffer:
        """Get a non-persistent memory buffer for anonymous users"""
        return ChatMemoryBuffer.from_defaults(
            token_limit=env_vars.CHAT_TOKEN_LIMIT,
            chat_store_key=f"anonymous:{session_id}:{session_type}",
        )

    async def create_session(
        self, user_email: str, session_data: ChatSessionCreate, session_id: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session for an authenticated user"""
        if not session_id:
            session_id = str(uuid.uuid4())

        now = datetime.now(timezone.utc)

        session = ChatSession(
            session_id=session_id,
            user_email=user_email,
            title=session_data.title,
            session_type=session_data.session_type,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

        try:
            # Store session metadata
            self.sessions_table.put_item(
                Item={
                    "session_id": session_id,
                    "user_email": user_email,
                    "title": session_data.title,
                    "session_type": session_data.session_type,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "message_count": 0,
                }
            )

            logger.info(f"Created session {session_id} for user {user_email}")
            return session

        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise e

    async def get_user_sessions(self, user_email: str, limit: int = 20, offset: int = 0) -> ChatSessionList:
        """Get all chat sessions for a user"""
        try:
            # Query sessions by user_email (assuming GSI exists)
            response = self.sessions_table.query(
                IndexName="user-email-index",  # You'll need to create this GSI
                KeyConditionExpression=Key("user_email").eq(user_email),
                ScanIndexForward=False,  # Sort by created_at descending
                Limit=limit + offset,
            )

            items = response.get("Items", [])

            # Apply offset
            items = items[offset:]

            sessions = []
            for item in items:
                session = ChatSession(
                    session_id=item["session_id"],
                    user_email=item["user_email"],
                    title=item["title"],
                    session_type=item["session_type"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    updated_at=datetime.fromisoformat(item["updated_at"]),
                    message_count=item.get("message_count", 0),
                )
                sessions.append(session)

            total = response.get("Count", 0)

            return ChatSessionList(sessions=sessions, total=total)

        except Exception as e:
            logger.error(f"Error getting user sessions: {str(e)}")
            raise e

    async def get_session(self, session_id: str, user_email: str) -> Optional[ChatSession]:
        """Get a specific session"""
        try:
            response = self.sessions_table.get_item(Key={"session_id": session_id})

            item = response.get("Item")
            if not item or item["user_email"] != user_email:
                return None

            return ChatSession(
                session_id=item["session_id"],
                user_email=item["user_email"],
                title=item["title"],
                session_type=item["session_type"],
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
                message_count=item.get("message_count", 0),
            )

        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            raise e

    async def update_session(
        self, session_id: str, user_email: str, update_data: ChatSessionUpdate
    ) -> Optional[ChatSession]:
        """Update a session's metadata"""
        try:
            update_expression = "SET updated_at = :updated_at"
            expression_values = {":updated_at": datetime.now(timezone.utc).isoformat()}

            if update_data.title:
                update_expression += ", title = :title"
                expression_values[":title"] = update_data.title

            response = self.sessions_table.update_item(
                Key={"session_id": session_id},
                UpdateExpression=update_expression,
                ConditionExpression="user_email = :user_email",
                ExpressionAttributeValues={**expression_values, ":user_email": user_email},
                ReturnValues="ALL_NEW",
            )

            item = response["Attributes"]
            return ChatSession(
                session_id=item["session_id"],
                user_email=item["user_email"],
                title=item["title"],
                session_type=item["session_type"],
                created_at=datetime.fromisoformat(item["created_at"]),
                updated_at=datetime.fromisoformat(item["updated_at"]),
                message_count=item.get("message_count", 0),
            )

        except Exception as e:
            logger.error(f"Error updating session: {str(e)}")
            raise e

    async def delete_session(self, session_id: str, user_email: str) -> bool:
        """Delete a session and all its messages"""
        try:
            # First verify the session belongs to the user
            session = await self.get_session(session_id, user_email)
            if not session:
                return False

            # Delete chat messages from the chat store
            chat_store_key = f"{user_email}:{session_id}:{session.session_type}"
            self.chat_store.delete_messages(chat_store_key)

            # Delete session metadata
            self.sessions_table.delete_item(
                Key={"session_id": session_id},
                ConditionExpression="user_email = :user_email",
                ExpressionAttributeValues={":user_email": user_email},
            )

            logger.info(f"Deleted session {session_id} for user {user_email}")
            return True

        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise e

    async def increment_message_count(self, session_id: str, user_email: str) -> None:
        """Increment the message count for a session"""
        try:
            self.sessions_table.update_item(
                Key={"session_id": session_id},
                UpdateExpression="ADD message_count :inc SET updated_at = :updated_at",
                ExpressionAttributeValues={
                    ":inc": 1,
                    ":updated_at": datetime.now(timezone.utc).isoformat(),
                    ":user_email": user_email,
                },
                ConditionExpression="user_email = :user_email",
            )
        except Exception as e:
            logger.error(f"Error incrementing message count: {str(e)}")
            # Don't raise - this is not critical

    async def get_session_messages(
        self, user_email: str, session_id: str, session_type: str = "mirag"
    ) -> List[ChatMessage]:
        """Get all chat messages for a specific session"""
        try:
            chat_store_key = f"{user_email}:{session_id}:{session_type}"

            # Get messages from the chat store
            messages = self.chat_store.get_messages(chat_store_key)

            return messages

        except Exception as e:
            logger.error(f"Error getting session messages: {str(e)}")
            return []

    async def get_session_history(self, user_email: str, session_id: str) -> Dict[str, List[ChatMessage]]:
        """Get chat history for both MiRAG and LongRAG for a session"""
        try:
            mirag_messages = await self.get_session_messages(user_email, session_id, "mirag")
            longrag_messages = await self.get_session_messages(user_email, session_id, "longrag")

            return {"mirag": mirag_messages, "longrag": longrag_messages}

        except Exception as e:
            logger.error(f"Error getting session history: {str(e)}")
            return {"mirag": [], "longrag": []}
