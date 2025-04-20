from typing import Iterable, List

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.workflow import Event, StartEvent, StopEvent

from utils.searxng import SearXNGClient


class PrepEvent(Event):
    """Prep event that signals the preparation phase for retrieval operations."""

    pass


class RetrieveEvent(Event):
    """Retrieve event containing nodes retrieved from the knowledge base.

    Attributes:
        retrieved_nodes: List of nodes with relevance scores.
    """

    retrieved_nodes: list[NodeWithScore]


class RelevanceEvalEvent(Event):
    """Relevance evaluation event with filtered results.

    Attributes:
        relevant_results: List of relevant search result strings.
    """

    relevant_results: list[str]


class TextExtractEvent(Event):
    """Text extract event with concatenated relevant information.

    Attributes:
        relevant_text: Concatenated text from relevant sources.
    """

    relevant_text: str


class QueryEvent(Event):
    """Query event for processing with context.

    Attributes:
        relevant_text: Text relevant to the query for context.
        search_text: The search query text.
    """

    relevant_text: str
    search_text: str


class LoadNodeEvent(Event):
    """Event for loading nodes into the retrieval system.

    Attributes:
        small_nodes: Iterable of individual TextNodes.
        grouped_nodes: List of grouped TextNodes.
        index: VectorStoreIndex for retrieval.
        similarity_top_k: Number of top similar nodes to retrieve.
        llm: Language model for processing.
    """

    small_nodes: Iterable[TextNode]
    grouped_nodes: list[TextNode]
    index: VectorStoreIndex
    similarity_top_k: int
    llm: LLM


class LongQueryStartEvent(StartEvent):
    """Start event for long-form query processing.

    Attributes:
        llm: Language model for processing.
        query_str: Query string to process.
        index: VectorStoreIndex for retrieval.
        history: List of previous interactions.
    """

    llm: LLM
    query_str: str
    index: VectorStoreIndex
    history: List
    memory: ChatMemoryBuffer


class LongQueryStopEvent(StopEvent):
    """Stop event for long-form query completion.

    Attributes:
        response: Generated streaming response.
    """

    response: StreamingAgentChatResponse


class MiRAGQueryStartEvent(StartEvent):
    """Start event for Mindful RAG query workflow.

    Attributes:
        llm: Language model for processing.
        query_str: Query string to process.
        index: VectorStoreIndex for retrieval.
        history: List of previous interactions.
        searxng: SearXNG client for web search.
    """

    llm: LLM
    query_str: str
    index: VectorStoreIndex
    history: List
    searxng: SearXNGClient
    memory: ChatMemoryBuffer


class ProcessRetrievedEvent(Event):
    """Event for processing an individual retrieved node.

    Attributes:
        node: Retrieved node with relevance score.
    """

    node: NodeWithScore


class ProgressEvent(Event):
    """Event for tracking progress in a workflow.

    Attributes:
        progress: Current step
    """

    progress: str
