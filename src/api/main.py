import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from starlette.middleware.sessions import SessionMiddleware

from api.auth import auth
from api.chat import chat
from api.dependencies import lifespan
from api.cli import CLI

from api.config import env_vars

# Load environment variables
load_dotenv()


# Initialize FastAPI app
app = FastAPI(title="MiRAG API", description="API for Mindful RAG Workflow", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.state.context.llm

app.add_middleware(SessionMiddleware, secret_key=env_vars.SECRET_KEY)
app.include_router(auth.router)
app.include_router(chat.router, prefix="/chat")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # global llm, index, searxng, index_manager, initialization_in_progress

    if app.state.initialization_in_progress:
        return {"status": "initializing", "message": "System is initializing components"}

    if app.state.index is None:
        return {"status": "not_ready", "message": "Index is not yet initialized"}

    return {"status": "ready", "message": "System is ready to process queries"}


def run():
    """Run the FastAPI application using uvicorn"""
    args = CLI.parse_arguments()
    logger.debug(args)
    logger.info(f"Starting MiRAG API server on {args.host}:{args.port}")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run()
