from re import L
from dotenv.main import load_dotenv
from icecream import ic
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Set, Any

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
)
from llama_index.core.node_parser import SentenceSplitter
from typing import Iterable
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.workflow import (
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    Document,
    PromptTemplate,
    SummaryIndex,
)

from utils.logger import logger
from llama_index.core.llms import LLM
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.base.base_retriever import BaseRetriever

from llama_index.core.readers import StringIterableReader
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode

from mirag.metrics import single_ans_em

load_dotenv()


# constants
DEFAULT_CHUNK_SIZE = 4096  # optionally splits documents into CHUNK_SIZE, then regroups them to demonstrate grouping algorithm
DEFAULT_MAX_GROUP_SIZE = 20  # maximum number of documents in a group
DEFAULT_SMALL_CHUNK_SIZE = 512  # small chunk size for generating embeddings
DEFAULT_TOP_K = 8  # top k for retrieving
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",
    embed_batch_size=64,
    cache_folder="./.embeddings",
    device="cuda",
)
# Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000)


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
    adj: Dict[str, List[str]] = {
        node.id_: [val.node_id for val in node.relationships.values()] for node in nodes
    }
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


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate the document's relevance.
    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)

EXTRACT_ANSWER = PromptTemplate(
    template="""Your task is to extract the answer from the given context and question. \n
    Context:
    \n ------- \n
    {context}
    \n ------- \n
    Question:
    \n ------- \n
    {question}
    \n ------- \n
    Your task is to derive a very concise short answer, extracting a substring from the given long answer.
    Respond with the answer only.
    If the answer has multiple parts, provide a single, coherent answer.
    It's important to ensure that the output short answer remains as simple as possible.
    Short answer is typically an entity without any other redundant words."""
)


class MindfulRAGWorkflow(Workflow):
    """Mindful RAG Workflow"""

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> LoadNodeEvent | None:
        """Ingestion step.

        Args:
            ctx (Context): Context object
            ev (StartEvent): StartEvent object

        Returns:
            LoadNodeEvent or None
        """
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
                nodes = split_doc(
                    chunk_size, docs
                )  # split documents into chunks of chunk_size
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

        # ic("ingest step")
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
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        tavily_ai_apikey: str | None = ev.get("tavily_ai_apikey")
        index = ev.get("index")

        llm = OpenAI(model="gpt-4o-mini")
        # llm = Gemini(model="models/gemini-1.5-flash")

        await ctx.set(
            "relevancy_pipeline",
            QueryPipeline(chain=[DEFAULT_RELEVANCY_PROMPT_TEMPLATE, llm]),
        )
        await ctx.set(
            "transform_query_pipeline",
            QueryPipeline(chain=[DEFAULT_TRANSFORM_QUERY_TEMPLATE, llm]),
        )
        # ic(llm)

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("tavily_tool", TavilyToolSpec(api_key=tavily_ai_apikey))

        await ctx.set("query_str", query_str)
        await ctx.set("retriever_kwargs", retriever_kwargs)

        # ic("prepare step")
        return PrepEvent()

    @step
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        retriever_kwargs = await ctx.get("retriever_kwargs")

        if query_str is None:
            return None

        index = await ctx.get("index", default=None)
        tavily_tool = await ctx.get("tavily_tool", default=None)
        if not (index or tavily_tool):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        # ic("in retrieve step", index)
        retriever: BaseRetriever = index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)

        logger.info(f"LongRAG retrieved {len(result)} documents")
        logger.info(f"LongRAG retrieval scores: {[node.score for node in result]}")

        await ctx.set("retrieved_nodes", result)
        await ctx.set("query_str", query_str)
        return RetrieveEvent(retrieved_nodes=result)

    @step
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        query_str = await ctx.get("query_str")

        relevancy_results = []
        for node in retrieved_nodes:
            relevancy_pipeline = await ctx.get("relevancy_pipeline")
            relevancy = relevancy_pipeline.run(
                context_str=node.text, query_str=query_str
            )
            relevancy_results.append(relevancy.message.content.lower().strip())

        # c("eval_relevance step", relevancy_results)

        relevancy_count = sum(1 for r in relevancy_results if r == "yes")
        logger.info(
            f"LongRAG context relevance: {relevancy_count}/{len(relevancy_results)} documents relevant"
        )

        await ctx.set("relevancy_results", relevancy_results)
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step
    async def extract_relevant_texts(
        self, ctx: Context, ev: RelevanceEvalEvent
    ) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        retrieved_nodes = await ctx.get("retrieved_nodes")
        relevancy_results = ev.relevant_results

        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]

        result = "\n".join(relevant_texts)
        # ic("extract_relevant_texts step", result)
        return TextExtractEvent(relevant_text=result)

    @step
    async def transform_query_pipeline(
        self, ctx: Context, ev: TextExtractEvent
    ) -> QueryEvent:
        """Search the transformed query with Tavily API."""
        relevant_text = ev.relevant_text
        relevancy_results = await ctx.get("relevancy_results")
        query_str = await ctx.get("query_str")

        # If any document is found irrelevant, transform the query string for better search results.
        if "no" in relevancy_results:
            logger.info(
                "LongRAG context insufficient - transforming query and using external search"
            )
            qp = await ctx.get("transform_query_pipeline")
            transformed_query_str = qp.run(query_str=query_str).message.content
            # Conduct a search with the transformed query string and collect the results.
            tavily_tool = await ctx.get("tavily_tool")
            search_results = tavily_tool.search(transformed_query_str, max_results=5)
            search_text = "\n".join([result.text for result in search_results])
        else:
            logger.info("LongRAG context fully relevant - no external search needed")
            search_text = ""

        # ic("transform_query_pipeline step", relevancy_results)
        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step
    async def metrics(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        llm = await ctx.get("llm")
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.get("query_str")
        relevancy_results = await ctx.get("relevancy_results")

        # Determine the status of the RAG process based on the relevancy results
        if all(result == "yes" for result in relevancy_results):
            status = "correct"
        elif not relevant_text and search_text:
            status = "incorrect"
        else:
            status = "ambiguous"

        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        result = query_engine.query(query_str)
        long_answer = str(result)
        # ic(result)

        qp = QueryPipeline(chain=[EXTRACT_ANSWER, llm])
        short_answer = qp.run(context=str(result), question=query_str)

        return StopEvent(
            result={
                "long_answer": long_answer,
                "short_answer": short_answer.message.content.lower().strip(),
                "status": status,
            }
        )


class LongRAGRetriever(BaseRetriever):
    """Long RAG Retriever."""

    def __init__(
        self,
        grouped_nodes: List[TextNode],
        small_toks: List[TextNode],
        vector_store: BasePydanticVectorStore,
        similarity_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Constructor.

        Args:
            grouped_nodes (List[TextNode]): Long retrieval units, nodes with docs grouped together based on relationships
            small_toks (List[TextNode]): Smaller tokens
            embed_model (BaseEmbedding, optional): Embed model. Defaults to None.
            similarity_top_k (int, optional): Similarity top k. Defaults to 8.
        """
        self._grouped_nodes = grouped_nodes
        self._grouped_nodes_dict = {node.id_: node for node in grouped_nodes}
        self._small_toks = small_toks
        self._small_toks_dict = {node.id_: node for node in self._small_toks}

        self._similarity_top_k = similarity_top_k
        self._vec_store = vector_store
        self._embed_model = Settings.embed_model

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieves.

        Args:
            query_bundle (QueryBundle): query bundle

        Returns:
            List[NodeWithScore]: nodes with scores
        """
        # make query
        query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding, similarity_top_k=500
        )

        # query for answer
        query_res = self._vec_store.query(vector_store_query)

        # determine top parents of most similar children (these are long retrieval units)
        top_parents_set: Set[str] = set()
        top_parents: List[NodeWithScore] = []
        for id_, similarity in zip(query_res.ids, query_res.similarities):
            cur_node = self._small_toks_dict[id_]
            parent_id = cur_node.ref_doc_id
            if parent_id not in top_parents_set:
                top_parents_set.add(parent_id)
                parent_node = self._grouped_nodes_dict[parent_id]
                node_with_score = NodeWithScore(node=parent_node, score=similarity)
                top_parents.append(node_with_score)

                if len(top_parents_set) >= self._similarity_top_k:
                    break

        assert len(top_parents) == min(self._similarity_top_k, len(self._grouped_nodes))

        return top_parents
