from pynamodb.models import Model
from api.config import env_vars
from pynamodb.indexes import GlobalSecondaryIndex, AllProjection
from pynamodb.attributes import UnicodeAttribute, NumberAttribute


class BaseTable(Model):
    class Meta:
        host = env_vars.DB_HOST if env_vars.ENVIRONMENT in ["dev", "test"] else None
        region = env_vars.AWS_REGION


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
        table_name = "user-table"

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
