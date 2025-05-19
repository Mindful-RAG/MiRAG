import asyncio
import json
import os
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

import requests
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Cookie, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from jose import ExpiredSignatureError, JWTError, jwt
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, StartEvent
from loguru import logger
from starlette.config import Config

# from api.auth.services import create_token, create_user
from api.config import env_vars

# from api.lib.utils import format_as_markdown
from api.models.query import LongragOut, MiragOut, QueryIn

# from api.models.user import UserIn
from mirag.events import LongQueryStartEvent, LongQueryStopEvent, MiRAGQueryStartEvent
from mirag.workflows import MindfulRAGWorkflow
from mirag.workflows_factory.simulation import MiragWorkflow, SimulationWorkflow

load_dotenv()

router = APIRouter()

# session_contexts = defaultdict(list)
longrag_session_contexts = defaultdict(list)
mirag_session_contexts = defaultdict(list)

# memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
# session_memories = defaultdict(lambda: ChatMemoryBuffer.from_defaults(token_limit=1500))
longrag_session_memories = defaultdict(
    lambda: ChatMemoryBuffer.from_defaults(token_limit=1500, chat_store_key="longrag")
)
mirag_session_memories = defaultdict(lambda: ChatMemoryBuffer.from_defaults(token_limit=1500, chat_store_key="mirag"))


@router.post("/mirag", response_model=MiragOut)
async def mirag_query(
    request: Request, query_request: QueryIn, background_tasks: BackgroundTasks
):  # , user=Depends(auth.get_current_user) remove for now
    """Process a query using the MindfulRAG workflow"""

    initialization_in_progress = request.app.state.initialization_in_progress
    index = request.app.state.index
    searxng = request.app.state.searxng
    llm = request.app.state.llm
    # wf = request.app.state.wf
    mirag: MiragWorkflow = request.app.state.mirag

    history = mirag_session_contexts[query_request.session_id]
    memory = mirag_session_memories[query_request.session_id]

    history.append(ChatMessage(role="user", content=query_request.query, additional_kwargs={"source": "mirag"}))
    # Check if initialization is in progress
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    # Process the query
    event = MiRAGQueryStartEvent(
        query_str=query_request.query, llm=llm, index=index["index"], searxng=searxng, history=history, memory=memory
    )
    run = mirag.run(start_event=event)

    async def response_stream():
        try:
            async for event in run.stream_events():
                if hasattr(event, "progress"):
                    yield json.dumps({"progress": event.progress}) + "\n"
                    await asyncio.sleep(0.01)

            wf = await run

            result = wf["response"]

            for token in result.response_gen:
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.005)

            mirag_session_contexts[query_request.session_id].append(
                ChatMessage(role="assistant", content=result.response, additional_kwargs={"source": "mirag"})
            )

            yield json.dumps({"done": "[DONE]"}) + "\n"
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            traceback.print_exc()  # Print the full stack trace for debugging
            yield json.dumps({"error": str(e)}) + "\n"
            yield json.dumps({"done": "[DONE]"}) + "\n"

    return StreamingResponse(
        response_stream(),
        media_type="application/json",
    )


@router.post("/longrag")
async def longrag_query(request: Request, query_request: QueryIn, background_tasks: BackgroundTasks):
    """Process a query using LongRAG and return a streaming response"""

    initialization_in_progress = request.app.state.initialization_in_progress
    index = request.app.state.index
    llm = request.app.state.llm
    longrag: SimulationWorkflow = request.app.state.longrag

    # Check if initialization is in progress
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    history = longrag_session_contexts[query_request.session_id]
    memory = longrag_session_memories[query_request.session_id]

    history.append(ChatMessage(role="user", content=query_request.query, additional_kwargs={"source": "longrag"}))

    start_e = LongQueryStartEvent(
        llm=llm, query_str=query_request.query, index=index["index"], history=history, memory=memory
    )
    run = longrag.run(start_event=start_e)

    async def response_stream():
        try:
            async for event in run.stream_events():
                if hasattr(event, "progress"):
                    yield json.dumps({"progress": event.progress}) + "\n"
                    await asyncio.sleep(0.01)

            wf = await run

            result = wf["response"]

            for token in result.response_gen:
                yield json.dumps({"token": token}) + "\n"
                await asyncio.sleep(0.005)

            longrag_session_contexts[query_request.session_id].append(
                ChatMessage(role="assistant", content=result.response, additional_kwargs={"source": "longrag"})
            )
            yield json.dumps({"done": "[DONE]"}) + "\n"
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            traceback.print_exc()  # Print the full stack trace for debugging
            yield json.dumps({"error": str(e)}) + "\n"
            yield json.dumps({"done": "[DONE]"}) + "\n"

    return StreamingResponse(
        response_stream(),
        media_type="application/json",
    )
