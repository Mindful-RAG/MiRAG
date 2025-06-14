import asyncio
from loguru import logger
from botocore.exceptions import ClientError
from api.db.tables import UserTable
from api.config import env_vars


async def initialize_dynamodb():
    """
    Initialize DynamoDB table if it doesn't exist
    """
    try:
        # Check if table exists
        if not UserTable.exists():
            logger.info("Creating DynamoDB table...")

            # Create table with on-demand billing
            UserTable.create_table(
                read_capacity_units=10,
                write_capacity_units=10,
                wait=True,  # Wait for table to be created
            )

            logger.info("DynamoDB table created successfully")
        else:
            logger.info("DynamoDB table already exists")

        # Test table access
        try:
            list(UserTable.scan(limit=1))
            logger.info("DynamoDB table is accessible")
        except Exception as e:
            logger.warning(f"DynamoDB table exists but may not be ready: {str(e)}")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error(f"DynamoDB initialization error: {error_code} - {str(e)}")

        if error_code == "UnauthorizedOperation":
            logger.error("AWS credentials may be invalid or insufficient permissions")
        elif error_code == "LimitExceededException":
            logger.error("AWS account limits exceeded")

        raise e
    except Exception as e:
        logger.error(f"Unexpected error initializing DynamoDB: {str(e)}")
        raise e
