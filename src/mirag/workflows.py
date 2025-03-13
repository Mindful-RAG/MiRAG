from typing import Any, Dict, FrozenSet, List, Optional, Set

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers import StringIterableReader
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from loguru import logger

from mirag.constants import DEFAULT_MAX_GROUP_SIZE
from mirag.events import LoadNodeEvent, PrepEvent, QueryEvent, RelevanceEvalEvent, RetrieveEvent, TextExtractEvent
from mirag.longrag_retriever import LongRAGRetriever
from mirag.prompts import (
    DEFAULT_RELEVANCY_PROMPT_TEMPLATE,
    DEFAULT_TRANSFORM_QUERY_TEMPLATE,
    EXTRACT_ANSWER,
    PREDICT_LONG_ANSWER,
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

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> LoadNodeEvent | None:
        """Ingestion step."""
        dataset: str | List[str] = ev.get("dataset")
        llm: LLM = ev.get("llm")
        chunk_size: int | None = ev.get("chunk_size")
        similarity_top_k: int = ev.get("similarity_top_k")
        small_chunk_size: int = ev.get("small_chunk_size")
        index: VectorStoreIndex | None = ev.get("index")
        index_kwargs: dict[str, Any] | None = ev.get("index_kwargs")

        if any(i is None for i in [dataset, llm, similarity_top_k, small_chunk_size]):
            return None
        if not index:
            docs = StringIterableReader().load_data(texts=dataset)
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

    @step
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

    @step
    async def prepare_for_retrieval(self, ctx: Context, ev: StartEvent) -> PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        context_titles: str | None = ev.get("context_titles")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})
        llm: LLM = ev.get("llm")
        searxng: SearXNGClient = ev.get("searxng")

        if query_str is None:
            return None

        index = ev.get("index")
        retriever = ev.get("retriever")

        # await ctx.set(
        #     "relevancy_pipeline",
        #     QueryPipeline(chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]),
        # )
        # await ctx.set(
        #     "transform_query_pipeline",
        #     QueryPipeline(chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]),
        # )
        # ic(llm)

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("retriever", retriever)
        await ctx.set("searxng", searxng)

        await ctx.set("query_str", query_str)
        await ctx.set("context_titles", context_titles)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        # ic("prepare step")
        return PrepEvent()

    @step
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")
        retriever = await ctx.get("retriever")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        searxng = await ctx.get("searxng", default=None)
        if not (index or searxng):
            raise ValueError(
                "Index and searxng must be constructed. Run with 'documents' and 'searxng_url' params first."
            )

        # ic("in retrieve step", index)
        retriever: LongRAGRetriever = index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)

        logger.debug(f"LongRAG retrieved {len(result)} documents")
        logger.debug(f"LongRAG retrieval scores: {[node.score for node in result]}")

        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step
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
                # relevancy_pipeline = await ctx.get("relevancy_pipeline")
                # relevancy = relevancy_pipeline.run(context_str=node.text, query_str=query_str)
                relevancy = await llm.apredict(
                    prompt=DEFAULT_RELEVANCY_PROMPT_TEMPLATE, centext_str=node.text, query_str=query_str
                )
                relevancy_results.append(relevancy)

        relevancy_count = sum(1 for r in relevancy_results if r == "yes")
        logger.debug(f"LongRAG context relevance: {relevancy_count}/{len(relevancy_results)} documents relevant")

        await ctx.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step
    async def extract_relevant_texts(self, ctx: Context, ev: RelevanceEvalEvent) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        retrieved_nodes = await ctx.get("retrieved_nodes")
        relevancy_results = ev.relevant_results

        relevant_texts = [retrieved_nodes[i].text for i, result in enumerate(relevancy_results) if result == "yes"]

        result = "\n".join(relevant_texts)
        # ic("extract_relevant_texts step", result)
        return TextExtractEvent(relevant_text=result)

    @step
    async def transform_query_pipeline(self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """Search the transformed query with SearXNG."""
        relevant_text = ev.relevant_text
        relevancy_results = await ctx.get("relevancy_results")
        query_str = await ctx.get("query_str")

        # If any document is found irrelevant, transform the query string for better search results.
        if "no" in relevancy_results:
            logger.debug("LongRAG context insufficient - transforming query and using external search")
            llm: LLM = await ctx.get("llm")
            # qp = await ctx.get("transform_query_pipeline"
            # transformed_query_str = qp.run(query_str=query_str).message.content
            transformed_query_str = await llm.apredict(prompt=DEFAULT_TRANSFORM_QUERY_TEMPLATE, query_str=query_str)
            # logger.debug(f"Transformed query string: {transformed_query_str}")

            # Conduct a search with the transformed query string and collect the results.
            searxng: SearXNGClient = await ctx.get("searxng")
            searxng_results = searxng.get_content_for_llm(query=transformed_query_str, max_results=5)

            # logger.debug(f"Search results: {search_results}")

            # search_text = "\n".join([result.text for result in search_results])
            search_text = "\n".join([result.content for result in searxng_results])
            # logger.debug(f"Search text: {search_text}")
        else:
            logger.debug("LongRAG context fully relevant - no external search needed")
            search_text = ""

        # ic("transform_query_pipeline step", relevancy_results)
        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step
    async def metrics(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        llm: LLM = await ctx.get("llm")
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.get("query_str")
        context_titles = await ctx.get("context_titles")
        relevancy_results = await ctx.get("relevancy_results")

        # Determine the status of the RAG process based on the relevancy results
        if all(result == "yes" for result in relevancy_results):
            status = "correct"
        elif not relevant_text and search_text:
            status = "incorrect"
        else:
            status = "ambiguous"

        # Prepend "web search" if search_text is present
        context_with_attribution = relevant_text
        if search_text:
            context_with_attribution = f"[Web Search]: {search_text}\n" + context_with_attribution

        long_answer = await llm.apredict(
            prompt=PREDICT_LONG_ANSWER,
            context_titles=context_titles,
            question=query_str,
            context=context_with_attribution,
        )

        short_answer = await llm.apredict(prompt=EXTRACT_ANSWER, question=query_str, long_answer=long_answer)

        return StopEvent(
            result={
                "long_answer": long_answer,
                "short_answer": short_answer,
                "status": status,
            }
        )
