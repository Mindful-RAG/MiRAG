import boto3
from botocore.exceptions import ClientError
from api.config import env_vars
from api.utils.observability import logger


def create_chat_store_tables():
    """Create DynamoDB tables for chat store if they don't exist"""

    dynamodb = boto3.client(
        "dynamodb",
        region_name=env_vars.AWS_REGION,
        aws_access_key_id=env_vars.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=env_vars.AWS_SECRET_ACCESS_KEY,
    )

    # Chat messages table (used by LlamaIndex DynamoDBChatStore)
    chat_table_name = env_vars.CHAT_STORE_TABLE_NAME

    # Chat sessions metadata table
    sessions_table_name = f"{env_vars.CHAT_STORE_TABLE_NAME}-sessions"

    try:
        # Create chat messages table
        dynamodb.create_table(
            TableName=chat_table_name,
            KeySchema=[
                {
                    "AttributeName": "SessionId",
                    "KeyType": "HASH",  # Partition key
                }
            ],
            AttributeDefinitions=[{"AttributeName": "SessionId", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        logger.info(f"Created chat messages table: {chat_table_name}")

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            logger.info(f"Chat messages table {chat_table_name} already exists")
        else:
            logger.error(f"Error creating chat messages table: {e}")
            raise

    try:
        # Create sessions metadata table
        dynamodb.create_table(
            TableName=sessions_table_name,
            KeySchema=[
                {
                    "AttributeName": "session_id",
                    "KeyType": "HASH",  # Partition key
                }
            ],
            AttributeDefinitions=[
                {"AttributeName": "session_id", "AttributeType": "S"},
                {"AttributeName": "user_email", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "user-email-index",
                    "KeySchema": [
                        {"AttributeName": "user_email", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        logger.info(f"Created sessions metadata table: {sessions_table_name}")

    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            logger.info(f"Sessions metadata table {sessions_table_name} already exists")
        else:
            logger.error(f"Error creating sessions metadata table: {e}")
            raise


if __name__ == "__main__":
    create_chat_store_tables()
