import json
import asyncio

from llama_index.core.llms import ChatMessage
from .observability import logger
import traceback


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
