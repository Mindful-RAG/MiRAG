from typing import Iterable

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.workflow import Event


class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""

    pass


class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: list[NodeWithScore]


class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""

    relevant_results: list[str]


class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""

    relevant_text: str


class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str


class LoadNodeEvent(Event):
    """Event for loading nodes."""

    small_nodes: Iterable[TextNode]
    grouped_nodes: list[TextNode]
    index: VectorStoreIndex
    similarity_top_k: int
    llm: LLM
