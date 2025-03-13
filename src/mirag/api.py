import os
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from loguru import logger
from pydantic import BaseModel

from mirag.index_management import IndexManager
from mirag.workflows import MindfulRAGWorkflow
from utils.searxng import SearXNGClient

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="MiRAG API", description="API for Mindful RAG Workflow")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
wf = MindfulRAGWorkflow(timeout=60)
llm = None
index = None
searxng = None
index_manager = None
initialization_in_progress = False


class QueryRequest(BaseModel):
    query: str
    llm_model: Optional[str] = "gpt-4o-mini"
    embed_model: Optional[str] = "BAAI/bge-large-en-v1.5"


class QueryResponse(BaseModel):
    query: str
    short_answer: str
    long_answer: str
    status: str
    markdown: str


def format_as_markdown(query: str, short_answer: str, long_answer: str, status: str) -> str:
    """Format the response as markdown for better display"""
    status_emoji = {"correct": "✅", "ambiguous": "⚠️", "incorrect": "❌", "error": "❗"}.get(status, "❓")

    markdown = f"""# Query Response

    ## Question
    {query}

    ## Short Answer
    {short_answer}

    ## Detailed Answer
    {long_answer}

    ---
    **Status**: {status} {status_emoji}
    """
    return markdown


async def initialize_components(llm_model: str = "gpt-4o-mini", embed_model: str = "BAAI/bge-large-en-v1.5"):
    """Initialize all necessary components for the workflow"""
    global llm, index, searxng, index_manager, initialization_in_progress

    initialization_in_progress = True
    try:
        # Configure embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model,
            embed_batch_size=64,
            cache_folder="./.embeddings",
            device="cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu",
        )

        # Configure LLM based on argument
        if "gpt" in llm_model:
            llm = OpenAI(model=llm_model, temperature=0)
            Settings.llm = llm
        elif "gemini" in llm_model:
            llm = Gemini(model=f"models/{llm_model}")
            Settings.llm = llm

        # Initialize SearXNG client
        searxng = SearXNGClient(instance_url=os.environ.get("SEARXNG_URL", "http://localhost:8080"))

        # Initialize index manager
        index_manager = IndexManager(os.environ.get("PERSIST_PATH", "./persisted_index"), wf, llm)

        # Load dataset and index - this would typically come from the dataset, but for API we'll load from disk
        class Args:
            load_index = True
            persist_index = False

        # Create a mock args object with just what we need
        args = Args()
        index = await index_manager.load_or_create_index(args)

        if index is None:
            logger.error("Failed to load index. Make sure you've run the CLI to create an index first.")
            raise ValueError("Index not available. Run the CLI first to create an index.")

        logger.info("Components initialized successfully")
    finally:
        initialization_in_progress = False


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    await initialize_components()


@app.post("/query", response_model=QueryResponse)
async def query(query_request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query using the MindfulRAG workflow"""
    global llm, index, searxng, initialization_in_progress

    # Check if initialization is in progress
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    # If the requested LLM model is different from the current one, reinitialize
    if (
        llm is None
        or (llm.__class__.__name__ == "OpenAI" and "gpt" not in query_request.llm_model)
        or (llm.__class__.__name__ == "Gemini" and "gemini" not in query_request.llm_model)
    ):
        # Don't run in background - wait for initialization to complete
        await initialize_components(query_request.llm_model, query_request.embed_model)

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    # Process the query
    try:
        result = await wf.run(
            query_str=query_request.query,
            context_titles=[],  # No context titles for direct API queries
            llm=llm,
            index=index["index"],
            searxng=searxng,
        )

        # Create markdown formatted version of the response
        markdown = format_as_markdown(
            query=query_request.query,
            short_answer=result["short_answer"],
            long_answer=result["long_answer"],
            status=result["status"],
        )

        return QueryResponse(
            query=query_request.query,
            short_answer=result["short_answer"],
            long_answer=result["long_answer"],
            status=result["status"],
            markdown=markdown,
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_markdown = format_as_markdown(
            query=query_request.query,
            short_answer="Error processing query",
            long_answer=f"An error occurred while processing your query: {str(e)}",
            status="error",
        )
        raise HTTPException(status_code=500, detail={"error": str(e), "markdown": error_markdown})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global initialization_in_progress, index

    if initialization_in_progress:
        return {"status": "initializing", "message": "System is initializing components"}

    if index is None:
        return {"status": "not_ready", "message": "Index is not yet initialized"}

    return {"status": "ready", "message": "System is ready to process queries"}


def run_api(host="0.0.0.0", port=8000, reload=False):
    """Run the FastAPI application using uvicorn"""
    logger.info(f"Starting MiRAG API server on {host}:{port}")
    uvicorn.run("mirag.api:app", host=host, port=port, reload=reload)
