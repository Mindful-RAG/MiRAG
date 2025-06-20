from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from api.db import initialize_dynamodb
from api.services.initialize_firebase import initialize_firebase
from api.services.chat.dynamodb_setup import create_chat_store_tables
from api.utils.observability import logger
import boto3

from llama_index.embeddings.openai import OpenAIEmbedding


from mirag.events import LongQueryStartEvent, MiRAGQueryStartEvent
from mirag.index_management import IndexManager
from mirag.workflows import MindfulRAGWorkflow
from mirag.workflows_factory.simulation import MiragWorkflow, LongRAGWorkflow
from utils.searxng import SearXNGClient
from .config import env_vars
from slowapi.util import get_remote_address
from slowapi import Limiter

# Load environment variables
load_dotenv()


@asynccontextmanager
@logger.catch
async def lifespan(app: FastAPI):
    """Initialize all necessary components for the workflow"""

    app.state.initialization_in_progress = True
    wf = MindfulRAGWorkflow(timeout=60, verbose=True)
    longrag = LongRAGWorkflow(timeout=60, verbose=True, start_event_class=LongQueryStartEvent)
    mirag = MiragWorkflow(timeout=60, verbose=True, start_event_class=MiRAGQueryStartEvent)
    openai_embedding = OpenAIEmbedding()
    s3 = boto3.client("s3")

    try:
        llm = OpenAI(model=env_vars.LLM_MODEL, temperature=0)
        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=env_vars.EMBED_MODEL,
            embed_batch_size=64,
            cache_folder="./.embeddings",
            device="mps",
        )

        # Initialize SearXNG client
        searxng = SearXNGClient(instance_url=env_vars.SEARXNG_URL)
        await searxng._test_connection()

        # Initialize Firebase
        initialize_firebase()
        await initialize_dynamodb()

        # Initialize DynamoDB tables for chat store
        try:
            create_chat_store_tables()
            logger.info("Chat store tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chat store tables: {e}")
            # Don't fail startup if chat tables can't be created
            # Chat persistence will be disabled but core functionality works

        # Initialize index manager
        index_manager = IndexManager(env_vars.PERSIST_PATH, wf, llm)

        # Load dataset and index - this would typically come from the dataset, but for API we'll load from disk
        class Args:
            load_index = True
            persist_index = False
            collection_name = env_vars.DEFAULT_CORPUS

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
        app.state.longrag = longrag
        app.state.mirag = mirag
        app.state.s3 = s3
        app.state.openai_embedding = openai_embedding
        app.state.custom_indexes = {}  # Dictionary to store custom indexes by corpus_id
        logger.info("Components initialized successfully")
    finally:
        app.state.initialization_in_progress = False
    yield
    await searxng.close()
