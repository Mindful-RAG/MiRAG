from pynamodb.models import Model
from pynamodb.indexes import GlobalSecondaryIndex, AllProjection
from pynamodb.attributes import UnicodeAttribute, NumberAttribute
from api.config import env_vars


class BaseTable(Model):
    class Meta:
        # Use DynamoDB Local for development, AWS DynamoDB for production
        host = env_vars.DB_HOST if env_vars.ENVIRONMENT in ["dev", "test"] and env_vars.DB_HOST else None
        region = env_vars.AWS_REGION

        # AWS credentials (only needed if not using IAM roles)
        if env_vars.AWS_ACCESS_KEY_ID and env_vars.AWS_SECRET_ACCESS_KEY:
            aws_access_key_id = env_vars.AWS_ACCESS_KEY_ID
            aws_secret_access_key = env_vars.AWS_SECRET_ACCESS_KEY


class UserEmailIndex(GlobalSecondaryIndex["UserTable"]):
    """
    Global secondary index for looking up by email
    """

    class Meta:
        index_name = "user-email-index"
        read_capacity_units = 10
        write_capacity_units = 10
        projection = AllProjection()

    pk = UnicodeAttribute(hash_key=True)
    sk = UnicodeAttribute(range_key=True)


class UserTable(BaseTable):
    class Meta(BaseTable.Meta):
        table_name = env_vars.TABLE_NAME or "user-table"

        # Billing mode for AWS (use on-demand or provisioned)
        billing_mode = "PAY_PER_REQUEST"  # or 'PROVISIONED'

        # For provisioned mode, set capacity units
        read_capacity_units = 10
        write_capacity_units = 10

    # Primary key attributes
    pk = UnicodeAttribute(hash_key=True)  # Partition key
    sk = UnicodeAttribute(range_key=True)  # Sort key

    # User attributes
    user_id = UnicodeAttribute(null=True)
    user_email = UnicodeAttribute(null=True)
    user_name = UnicodeAttribute(null=True)
    user_pic = UnicodeAttribute(null=True)
    first_logged_in = NumberAttribute(null=True)
    last_accessed = NumberAttribute(null=True)

    # Token attributes
    access_token = UnicodeAttribute(null=True)
    session_id = UnicodeAttribute(null=True)

    # Query attributes
    query_id = UnicodeAttribute(null=True)
    answer = UnicodeAttribute(null=True)

    # Create GSI
    email_index = UserEmailIndex()
