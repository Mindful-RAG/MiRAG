from firebase_admin import initialize_app, credentials, get_app
from api.config import env_vars

from api.utils.observability import logger


@logger.catch
def initialize_firebase():
    """
    Initialize Firebase Admin SDK with credentials from settings.
    This function is safe to call multiple times as it will only initialize Firebase once.
    """
    try:
        try:
            get_app()
            logger.info("Firebase already initialized")
            return True
        except ValueError:
            # If Firebase is not yet initialized, proceed with initialization
            pass

        cred_path = getattr(env_vars, "GOOGLE_APPLICATION_CREDENTIALS", None)

        if not cred_path:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS not found in settings")
            return False

        # Initialize Firebase with the credentials
        cred = credentials.Certificate(cred_path)
        initialize_app(cred)

        logger.info("Firebase successfully initialized")
        return True

    except Exception as e:
        logger.error(f"Error initializing Firebase: {str(e)}", exc_info=True)
        return False
