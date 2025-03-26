from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from loguru import logger


from mirag.index_management import IndexManager
from mirag.workflows import MindfulRAGWorkflow
from utils.searxng import SearXNGClient
from .config import env_vars

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all necessary components for the workflow"""

    app.state.initialization_in_progress = True
    wf = MindfulRAGWorkflow(timeout=60, verbose=True)

    # Configure embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=env_vars.EMBED_MODEL,
        embed_batch_size=64,
        cache_folder="./.embeddings",
        device="cuda",
    )

    try:
        llm = OpenAI(model=env_vars.LLM_MODEL, temperature=0)
        Settings.llm = llm

        # Initialize SearXNG client
        searxng = SearXNGClient(instance_url=env_vars.SEARXNG_URL)
        await searxng._test_connection()

        # Initialize index manager
        index_manager = IndexManager(env_vars.PERSIST_PATH, wf, llm)

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

        app.state.llm = llm
        app.state.searxng = searxng
        app.state.index = index
        app.state.wf = wf
        logger.info("Components initialized successfully")
    finally:
        app.state.initialization_in_progress = False
    yield
    await searxng.close()
