import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware

from api.auth import auth
from api.auth.middleware import AuthMiddleware
from api.chat import chat
from api.cli import CLI
from api.config import env_vars
from api.dependencies import lifespan
from api.utils.observability import logger

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app = FastAPI(title="MiRAG API", description="API for Mindful RAG Workflow", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=env_vars.ALLOWED_ORIGINS if env_vars.ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"Allowed origins: {env_vars.ALLOWED_ORIGINS if env_vars.ENVIRONMENT == 'production' else ['*']}")

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key=env_vars.SECRET_KEY)

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/health")
@logger.catch
@limiter.exempt
async def health_check():
    """Health check endpoint"""

    if app.state.initialization_in_progress:
        return JSONResponse(
            status_code=503, content={"status": "initializing", "message": "System is initializing components"}
        )

    if app.state.index is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "message": "Index is not yet initialized"})

    return JSONResponse(status_code=200, content={"status": "ready", "message": "System is ready to process queries"})


def run():
    """Run the FastAPI application using uvicorn"""
    args = CLI.parse_arguments()
    logger.debug(args)
    logger.info(f"Starting MiRAG API server on {args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload, log_config=None, log_level=None)


if __name__ == "__main__":
    run()
