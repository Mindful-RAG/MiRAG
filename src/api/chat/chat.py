import asyncio
import io
import json
import traceback
import uuid
from collections import defaultdict

from botocore.exceptions import BotoCoreError, ClientError
from datasets.config import UPLOADS_MAX_NUMBER_PER_COMMIT
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from s3fs import S3FileSystem

from api.auth.services import OptionalUserDependency, UserDependency

# from api.auth.services import create_token, create_user
from api.config import env_vars

# from api.lib.utils import format_as_markdown
from api.models.query import MiragOut, QueryIn, UploadResponse
from api.services.llamaindex.corpus import IndexCorpus
from api.utils.observability import logger

# from api.models.user import UserIn
from mirag.events import LongQueryStartEvent, MiRAGQueryStartEvent
from mirag.workflows_factory.simulation import LongRAGWorkflow, MiragWorkflow

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


async def response_stream(run, query_request, source, session_context):
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

        session_context[query_request.session_id].append(
            ChatMessage(role="assistant", content=result.response, additional_kwargs={"source": source})
        )
        yield json.dumps({"done": "[DONE]"}) + "\n"
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        traceback.print_exc()  # Print the full stack trace for debugging
        yield json.dumps({"error": str(e)}) + "\n"
        yield json.dumps({"done": "[DONE]"}) + "\n"


async def get_index_for_query(request: Request, query_request: QueryIn):
    """Get the appropriate index based on the query request"""
    if query_request.custom_corpus_id:
        # Check if custom index exists
        custom_indexes = request.app.state.custom_indexes
        logger.info(f"Available custom indexes: {list(custom_indexes.keys())}")
        if query_request.custom_corpus_id in custom_indexes:
            logger.info(f"Using custom index: {query_request.custom_corpus_id}")
            return custom_indexes[query_request.custom_corpus_id]
        else:
            logger.warning(f"Custom corpus {query_request.custom_corpus_id} not found, using default index")

    # Return default index
    return request.app.state.index


@logger.catch
@router.post("/mirag", response_model=MiragOut)
async def mirag_query(
    request: Request, query_request: QueryIn, current_user: OptionalUserDependency
):  # , user=Depends(auth.get_current_user) remove for now
    """Process a query using the MindfulRAG workflow"""
    initialization_in_progress = request.app.state.initialization_in_progress
    searxng = request.app.state.searxng
    llm = request.app.state.llm
    mirag: MiragWorkflow = request.app.state.mirag
    assert isinstance(mirag, MiragWorkflow)

    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    # Get the appropriate index (default or custom)
    index = await get_index_for_query(request, query_request)

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    if current_user:
        logger.debug(f"user {current_user['user_email']} initiated LongRAG query: {query_request.query}")
    else:
        logger.info({"message": "Hello anonymous user"})

    history = mirag_session_contexts[query_request.session_id]
    memory = mirag_session_memories[query_request.session_id]

    history.append(ChatMessage(role="user", content=query_request.query, additional_kwargs={"source": "mirag"}))

    event = MiRAGQueryStartEvent(
        query_str=query_request.query, llm=llm, index=index["index"], searxng=searxng, history=history, memory=memory
    )
    run = mirag.run(start_event=event)

    return StreamingResponse(
        response_stream(run=run, query_request=query_request, source="mirag", session_context=mirag_session_contexts),
        media_type="application/json",
    )


@logger.catch
@router.post("/longrag")
async def longrag_query(request: Request, query_request: QueryIn, current_user: OptionalUserDependency):
    """Process a query using LongRAG and return a streaming response"""

    initialization_in_progress = request.app.state.initialization_in_progress
    llm = request.app.state.llm
    longrag: LongRAGWorkflow = request.app.state.longrag

    assert isinstance(longrag, LongRAGWorkflow)

    if current_user:
        logger.debug(f"user {current_user['user_email']} initiated LongRAG query: {query_request.query}")
    else:
        logger.debug({"message": "Hello anonymous user"})
    if initialization_in_progress:
        raise HTTPException(status_code=503, detail="System is initializing. Please try again in a moment.")

    # Get the appropriate index (default or custom)
    index = await get_index_for_query(request, query_request)

    logger.info(index)

    if not index:
        raise HTTPException(status_code=503, detail="Index is not yet initialized. Please try again later.")

    history = longrag_session_contexts[query_request.session_id]
    memory = longrag_session_memories[query_request.session_id]

    history.append(ChatMessage(role="user", content=query_request.query, additional_kwargs={"source": "longrag"}))

    start_e = LongQueryStartEvent(
        llm=llm, query_str=query_request.query, index=index["index"], history=history, memory=memory
    )
    run = longrag.run(start_event=start_e)

    return StreamingResponse(
        response_stream(
            run=run, query_request=query_request, source="longrag", session_context=longrag_session_contexts
        ),
        media_type="application/json",
    )


@logger.catch
@router.post("/upload", response_model=UploadResponse)
async def upload_file(request: Request, current_user: UserDependency, file: UploadFile = File(...)):
    """Upload a PDF file to the server to be indexed"""
    s3 = request.app.state.s3
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()

    if len(contents) > env_vars.MAX_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10 MB")

    try:
        s3.upload_fileobj(
            io.BytesIO(contents), env_vars.BUCKET_NAME, file.filename, ExtraArgs={"ContentType": file.content_type}
        )
    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

    try:
        s3_fs = S3FileSystem(anon=False)

        def get_metadata(file_name):
            return {
                "file_name": file_name,
                "file_type": file.content_type,
                "size": file.size,
                "user_email": current_user["user_email"],
            }

        reader = SimpleDirectoryReader(
            input_files=[f"{env_vars.BUCKET_NAME}/{file.filename}"],
            fs=s3_fs,
            file_metadata=get_metadata,
        )
        docs = reader.load_data()
        index_corpus = IndexCorpus(
            wf=request.app.state.wf,
            embed_model=request.app.state.openai_embedding,
            llm=request.app.state.llm,
        )

        class Args:
            load_index = False
            persist_index = False
            collection_name = str(uuid.uuid4())

        args = Args()
        custom_index = await index_corpus.index_corpus(args, docs)
        logger.info(custom_index)

        corpus_id = args.collection_name
        request.app.state.custom_indexes[corpus_id] = custom_index

        logger.info(f"Created custom index with corpus_id: {corpus_id} for user: {current_user['user_email']}")

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    return UploadResponse(file=file.filename, content_type=file.content_type, size=file.size, corpus_id=corpus_id)


@logger.catch
@router.delete("/upload/{corpus_id}")
async def delete_custom_corpus(request: Request, current_user: UserDependency, corpus_id: str):
    """Delete a custom corpus index"""
    custom_indexes = request.app.state.custom_indexes

    if corpus_id not in custom_indexes:
        raise HTTPException(status_code=404, detail="Custom corpus not found")

    # Remove the custom index from memory
    del custom_indexes[corpus_id]

    logger.info(f"Deleted custom index {corpus_id} for user: {current_user['user_email']}")

    return {"message": "Custom corpus deleted successfully", "corpus_id": corpus_id}
