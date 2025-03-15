import asyncio
from typing import Any, Coroutine


def asyncio_run(coro: Coroutine) -> Any:
    """Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # If we're here, there's an existing loop but it's not running
        return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread, or its already running
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )
