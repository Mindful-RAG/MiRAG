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

from mirag.constants import DEFAULT_MAX_GROUP_SIZE, DEFAULT_TOP_K
from mirag.events import LoadNodeEvent, PrepEvent, QueryEvent, RelevanceEvalEvent, RetrieveEvent, TextExtractEvent
from mirag.hf_document import hf_dataset_to_documents
from mirag.longrag_retriever import LongRAGRetriever
from mirag.prompts import (
    DEFAULT_RELEVANCY_PROMPT_TEMPLATE,
    DEFAULT_TRANSFORM_QUERY_TEMPLATE,
    EXTRACT_ANSWER,
    PREDICT_LONG_ANSWER_NQ,
    PREDICT_LONG_ANSWER_QA,
)
from utils.searxng import SearXNGClient

load_dotenv()


def split_doc(chunk_size: int, documents: List[BaseNode]) -> List[TextNode]:
    """Splits documents into smaller pieces.

    Args:
        chunk_size (int): Chunk size
        documents (List[BaseNode]): Documents

    Returns:
        List[TextNode]: Smaller chunks
    """
    # split docs into tokens
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    return text_parser.get_nodes_from_documents(documents)


def group_docs(
    nodes: List[str],
    adj: Dict[str, List[str]],
    max_group_size: Optional[int] = DEFAULT_MAX_GROUP_SIZE,
) -> Set[FrozenSet[str]]:
    """Groups documents.

    Args:
        nodes (List[str]): documents IDs
        adj (Dict[str, List[str]]): related documents for each document; id -> list of doc strings
        max_group_size (Optional[int], optional): max group size, None if no max group size. Defaults to DEFAULT_MAX_GROUP_SIZE.
    """
    docs = sorted(nodes, key=lambda node: len(adj[node]))
    groups = set()  # set of set of IDs
    for d in docs:
        related_groups = set()
        for r in adj[d]:
            for g in groups:
                if r in g:
                    related_groups = related_groups.union(frozenset([g]))

        gnew = {d}
        related_groupsl = sorted(related_groups, key=lambda el: len(el))
        for g in related_groupsl:
            if max_group_size is None or len(gnew) + len(g) <= max_group_size:
                gnew = gnew.union(g)
                if g in groups:
                    groups.remove(g)

        groups.add(frozenset(gnew))

    return groups


def get_grouped_docs(
    nodes: List[TextNode],
    max_group_size: Optional[int] = DEFAULT_MAX_GROUP_SIZE,
) -> List[TextNode]:
    """Gets list of documents that are grouped.

    Args:
        nodes (t.List[TextNode]): Input list
        max_group_size (Optional[int], optional): max group size, None if no max group size. Defaults to DEFAULT_MAX_GROUP_SIZE.

    Returns:
        t.List[TextNode]: Output list
    """
    # node IDs
    nodes_str = [node.id_ for node in nodes]
    # maps node ID -> related node IDs based on that node's relationships
    adj: Dict[str, List[str]] = {node.id_: [val.node_id for val in node.relationships.values()] for node in nodes}
    # node ID -> node
    nodes_dict = {node.id_: node for node in nodes}

    res = group_docs(nodes_str, adj, max_group_size)

    ret_nodes = []
    for g in res:
        cur_node = TextNode()

        for node_id in g:
            cur_node.text += nodes_dict[node_id].text + "\n\n"
            cur_node.metadata.update(nodes_dict[node_id].metadata)

        ret_nodes.append(cur_node)

    return ret_nodes


class MindfulRAGWorkflow(Workflow):
    """Mindful RAG Workflow"""

    @step(num_workers=16)
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
            if chunk_size is not None:
                nodes = split_doc(chunk_size, docs)  # split documents into chunks of chunk_size
                grouped_nodes = get_grouped_docs(
                    nodes
                )  # get list of nodes after grouping (groups are combined into one node), these are long retrieval units
            else:
                grouped_nodes = docs

            # split large retrieval units into smaller nodes
            small_nodes = split_doc(small_chunk_size, grouped_nodes)

            index_kwargs = index_kwargs or {}
            index = VectorStoreIndex(small_nodes, **index_kwargs)
        else:
            # get smaller nodes from index and form large retrieval units from these nodes
            small_nodes = index.docstore.docs.values()
            grouped_nodes = get_grouped_docs(small_nodes, None)

        return LoadNodeEvent(
            small_nodes=small_nodes,
            grouped_nodes=grouped_nodes,
            index=index,
            similarity_top_k=similarity_top_k,
            llm=llm,
        )

    @step(pass_context=True)
    async def make_query_engine(self, ctx: Context, ev: LoadNodeEvent) -> StopEvent:
        """Query engine construction step.

        Args:
            ctx (Context): context
            ev (LoadNodeEvent): event

        Returns:
            StopEvent: stop event
        """

        # make retriever and query engine
        retriever = LongRAGRetriever(
            grouped_nodes=ev.grouped_nodes,
            small_toks=ev.small_nodes,
            similarity_top_k=ev.similarity_top_k,
            vector_store=ev.index.vector_store,
        )
        query_eng = RetrieverQueryEngine.from_args(retriever, ev.llm)

        # ic("make query engine step")
        return StopEvent(
            result={
                "retriever": retriever,
                "query_engine": query_eng,
                "index": ev.index,
            }
        )

    @step(pass_context=True)
    async def prepare_for_retrieval(self, ctx: Context, ev: StartEvent) -> PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        context_titles: str | None = ev.get("context_titles")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})
        llm: LLM = ev.get("llm")
        searxng: SearXNGClient = ev.get("searxng")
        data_name: str = ev.get("data_name")

        if query_str is None:
            return None

        index = ev.get("index")

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("searxng", searxng)
        await ctx.set("data_name", data_name)

        await ctx.set("query_str", query_str)
        await ctx.set("context_titles", context_titles)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        return PrepEvent()

    @step(pass_context=True, num_workers=8)
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index: VectorStoreIndex = await ctx.get("index", default=None)
        searxng = await ctx.get("searxng", default=None)
        if not (index or searxng):
            raise ValueError(
                "Index and searxng must be constructed. Run with 'documents' and 'searxng_url' params first."
            )

        # ic("in retrieve step", index)
        # getretrieve = retrieve.retrieve(query_str)
        # logger.debug(getretrieve)
        # retriever = index.as_retriever(**retriever_kwargs)
        retriever = index.as_retriever(similarity_top_k=8)
        # result = retrieve.retrieve(query_str)
        result = retriever.retrieve(query_str)

        logger.debug(f"LongRAG retrieved {len(result)} documents")
        logger.debug(f"LongRAG retrieval scores: {[node.score for node in result]}")

        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step(pass_context=True)
    async def eval_relevance(self, ctx: Context, ev: RetrieveEvent) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.get("query_str")
        relevancy_score_threshold: int = await ctx.get("relevancy_score_threshold", default=0.7)

        relevancy_results = []
        for node in retrieved_nodes:
            # Automatically consider documents with high similarity scores as relevant
            if hasattr(node, "score") and node.score is not None and node.score >= relevancy_score_threshold:
                relevancy_results.append("yes")
                logger.debug(f"Document with score {node.score} automatically marked as relevant")
            else:
                # Only use LLM evaluation for documents below the threshold
                llm: LLM = await ctx.get("llm")
                relevancy = llm.complete(
                    prompt=DEFAULT_RELEVANCY_PROMPT_TEMPLATE.format(
                        context_str=node.text, metadata=node.metadata, query_str=query_str
                    )
                )
                logger.debug(relevancy)
                # relevancy_results.append(relevancy.message.content.lower().strip())
                # relevancy_results.append(relevancy)
                relevancy_results.append(relevancy.text.lower().strip())
                logger.debug(relevancy_results)

        relevancy_count = sum(1 for r in relevancy_results if r == "yes")
        logger.debug(f"LongRAG context relevance: {relevancy_count}/{len(relevancy_results)} documents relevant")

        await ctx.set("relevancy_score", relevancy_count / len(relevancy_results))
        await ctx.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step(pass_context=True)
    async def extract_relevant_texts(self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        retrieved_nodes = await ctx.get("retrieved_nodes")
        relevancy_results = ev.relevant_results

        relevant_texts = [retrieved_nodes[i].text for i, result in enumerate(relevancy_results) if result == "yes"]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)

    @step(pass_context=True)
    async def transform_query_pipeline(self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """Search the transformed query with SearXNG."""
        relevant_text = ev.relevant_text
        # relevancy_results = await ctx.get("relevancy_results")
        query_str = await ctx.get("query_str")
        relevancy_score = await ctx.get("relevancy_score")
        # If any document is found irrelevant, transform the query string for better search results.
        # if "no" in relevancy_results:
        if relevancy_score < 0.5:
            logger.debug("LongRAG context insufficient - transforming query and using external search")
            llm: LLM = await ctx.get("llm")
            transformed_query_str = await llm.acomplete(
                prompt=DEFAULT_TRANSFORM_QUERY_TEMPLATE.format(query_str=query_str)
            )
            logger.debug(f"Transformed query string: {transformed_query_str}")

            # Conduct a search with the transformed query string and collect the results.
            searxng: SearXNGClient = await ctx.get("searxng")
            searxng_results = await searxng.get_content_for_llm(query=transformed_query_str.text, max_results=10)

            search_text = "\n".join([result.content for result in searxng_results])
        else:
            logger.debug("LongRAG context fully relevant - no external search needed")
            search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step(pass_context=True)
    async def metrics(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        llm: LLM = await ctx.get("llm")
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.get("query_str")
        context_titles = await ctx.get("context_titles")
        relevancy_score = await ctx.get("relevancy_score")
        data_name = await ctx.get("data_name")

        # Determine the status of the RAG process based on the relevancy results
        if relevancy_score > 0.5:
            status = "correct"
        elif relevancy_score == 0.0:
            status = "incorrect"
        else:
            status = "ambiguous"

        # Prepend "web search" if search_text is present
        context_with_attribution = f"[Document]: {relevant_text}\n\n[Web Search]: {search_text}"

        # long_answer = ""
        # if data_name == "hotpot_qa":
        #     long_answer_completion = await llm.acomplete(
        #         prompt=PREDICT_LONG_ANSWER_QA.format(
        #             titles=context_titles, question=query_str, context=context_with_attribution
        #         ),
        #     )
        #     long_answer = long_answer_completion.text
        # else:
        long_answer_completion = await llm.acomplete(
            prompt=PREDICT_LONG_ANSWER_NQ.format(
                titles=context_titles, question=query_str, context=context_with_attribution
            ),
        )
        long_answer = long_answer_completion.text

        short_answer = await llm.acomplete(
            prompt=EXTRACT_ANSWER.format(long_answer=long_answer, question=query_str),
        )

        return StopEvent(
            result={
                "long_answer": long_answer,
                "short_answer": short_answer.text,
                "status": status,
            }
        )

    @step
    async def longrag_query(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Query step.

        Args:
            ctx (Context): context
            ev (StartEvent): start event

        Returns:
            StopEvent | None: stop event with result
        """
        llm: LLM = ev.get("long_llm")
        query_str: str | None = ev.get("long_query_str")
        index: VectorStoreIndex = ev.get("long_index")

        if query_str is None:
            return None

        retriever = index.as_query_engine(retriever_mode="llm", choice_batch_size=5)
        long_answer = await retriever.aquery(query_str)

        short_answer = await llm.acomplete(
            prompt=EXTRACT_ANSWER.format(long_answer=str(long_answer), question=query_str),
        )

        return StopEvent(
            result={
                "long_answer": str(long_answer),
                "short_answer": short_answer.text,
            }
        )
