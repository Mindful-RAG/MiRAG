from typing import Any, Dict, FrozenSet, List, Optional, Set

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import LLM, CompletionResponse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers.string_iterable import StringIterableReader
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import DocumentStore, SimpleDocumentStore
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from loguru import logger

from mirag.events import LoadNodeEvent


class BaseWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> LoadNodeEvent | None:
        """Ingestion step."""
        dataset: List[str] | List[Document] = ev.get("dataset")
        llm: LLM = ev.get("llm")
        chunk_size: int | None = ev.get("chunk_size")
        similarity_top_k: int = ev.get("similarity_top_k")
        small_chunk_size: int = ev.get("small_chunk_size")
        index: VectorStoreIndex | None = ev.get("index")
        index_kwargs: dict[str, Any] | None = ev.get("index_kwargs")

        if any(i is None for i in [dataset, llm, similarity_top_k, small_chunk_size]):
            return None
        if not index:
            # docs = StringIterableReader().load_data(texts=dataset)
            docs = hf_dataset_to_documents(
                dataset=dataset, text_field="context", metadata_fields=["context_titles"]
            )  # metadata_fields
            logger.debug(dir(docs[0]))
            if chunk_size is not None:
                nodes = split_doc(chunk_size, docs)  # split documents into chunks of chunk_size
                grouped_nodes = get_grouped_docs(
                    nodes
                )  # get list of nodes after grouping (groups are combined into one node), these are long retrieval units
            else:
                grouped_nodes = docs

            # split large retrieval units into smaller nodes
            small_nodes = split_doc(small_chunk_size, grouped_nodes)

            logger.debug("ran in if")
            index_kwargs = index_kwargs or {}
            index = VectorStoreIndex(small_nodes, **index_kwargs)
        else:
            # get smaller nodes from index and form large retrieval units from these nodes
            small_nodes = index.docstore.docs.values()
            grouped_nodes = get_grouped_docs(small_nodes, None)
            logger.debug("ran in else")

        return LoadNodeEvent(
            small_nodes=small_nodes,
            grouped_nodes=grouped_nodes,
            index=index,
            similarity_top_k=similarity_top_k,
            llm=llm,
        )
