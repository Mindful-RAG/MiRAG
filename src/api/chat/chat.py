import os
import traceback
import uuid
from datetime import datetime, timedelta

import requests
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import APIRouter, Cookie, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from jose import ExpiredSignatureError, JWTError, jwt
from loguru import logger
from starlette.config import Config
# from api.lib.utils import format_as_markdown
from api.models.query import QueryIn, LongragOut, MiragOut

from fastapi import BackgroundTasks, FastAPI, HTTPException

# from api.auth.services import create_token, create_user
from api.config import env_vars
# from api.models.user import UserIn
from mirag.workflows import MindfulRAGWorkflow

load_dotenv()

router = APIRouter()


@router.post("/mirag", response_model=MiragOut)
async def mirag_query(
    request: Request, query_request: QueryIn, background_tasks: BackgroundTasks
):  # , user=Depends(auth.get_current_user) remove for now
    """Process a query using the MindfulRAG workflow"""

    initialization_in_progress = request.app.state.initialization_in_progress
    index = request.app.state.index
    searxng = request.app.state.searxng
    llm = request.app.state.llm
    wf = request.app.state.wf

    # Check if initialization is in progress
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    # Process the query
    try:
        result = await wf.run(
            query_str=query_request.query,
            context_titles=[],  # No context titles for direct API queries
            llm=llm,
            index=index["index"],
            data_name=env_vars.DATA_NAME,
            searxng=searxng,
        )
        logger.debug(result)

        # Create markdown formatted version of the response
        # markdown = format_as_markdown(
        #     query=query_request.query,
        #     short_answer=result["short_answer"],
        #     long_answer=result["long_answer"],
        #     status=result["status"],
        # )

        return MiragOut(
            query=query_request.query,
            short_answer=result["short_answer"],
            long_answer=result["long_answer"],
            status=result["status"],
            markdown="",
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # error_markdown = format_as_markdown(
        #     query=query_request.query,
        #     short_answer="Error processing query",
        #     long_answer=f"An error occurred while processing your query: {str(e)}",
        #     status="error",
        # )
        raise HTTPException(status_code=500, detail={"error": str(e), "markdown": ""})


@router.post("/longrag", response_model=LongragOut)
async def longrag_query(
    request: Request, query_request: QueryIn, background_tasks: BackgroundTasks
):  # , user=Depends(auth.get_current_user) remove for now
    """Process a query using the MindfulRAG workflow"""

    initialization_in_progress = request.app.state.initialization_in_progress
    index = request.app.state.index
    llm = request.app.state.llm
    wf = request.app.state.wf

    # Check if initialization is in progress
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    # Process the query
    try:
        result = await wf.run(
            long_query_str=query_request.query,
            long_llm=llm,
            long_index=index["index"],
        )
        logger.debug(result)

        # Create markdown formatted version of the response
        # markdown = format_as_markdown(
        #     query=query_request.query,
        #     short_answer=result["short_answer"],
        #     long_answer=result["long_answer"],
        #     status="",
        # )

        return LongragOut(
            query=query_request.query,
            short_answer=result["short_answer"],
            long_answer=result["long_answer"],
            markdown="",
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # error_markdown = format_as_markdown(
        #     query=query_request.query,
        #     short_answer="Error processing query",
        #     long_answer=f"An error occurred while processing your query: {str(e)}",
        #     status="error",
        # )
        raise HTTPException(status_code=500, detail={"error": str(e), "markdown": ""})
