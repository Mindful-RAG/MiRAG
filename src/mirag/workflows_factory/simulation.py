from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import step
from mirag.longrag_retriever import LongRAGRetriever
from mirag.prompts import (
    DEFAULT_RELEVANCY_PROMPT_TEMPLATE,
    DEFAULT_TRANSFORM_QUERY_TEMPLATE,
    EVALUATE_QUERY,
    EXTRACT_ANSWER,
    PREDICT_ANSWER,
    PREDICT_LONG_ANSWER_NQ,
)
from mirag.workflows_factory.base import BaseWorkflow

from typing import Any, Dict, FrozenSet, List, Optional, Set

from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import LLM, ChatMessage, CompletionResponse
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

from mirag.events import (
    LongQueryStartEvent,
    LongQueryStopEvent,
    MiRAGQueryStartEvent,
    ProcessRetrievedEvent,
    ProgressEvent,
)
from mirag.events import LoadNodeEvent, PrepEvent, QueryEvent, RelevanceEvalEvent, RetrieveEvent, TextExtractEvent
from utils.searxng import SearXNGClient


# refactor this, ugly af
class SimulationWorkflow(Workflow):
    @step
    async def query_longrag(self, ctx: Context, ev: LongQueryStartEvent) -> StopEvent:
        """Query step.

        Args:
            ctx (Context): context
            ev (StartEvent): start event

        Returns:
            LongQueryStopEvent: stop event with result
        """
        llm: LLM = ev.llm
        query_str: str = ev.query_str
        index: VectorStoreIndex = ev.index
        history: List = ev.history
        memory: ChatMemoryBuffer = ev.memory

        # memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        # retriever = index.as_chat_engine(llm=llm, chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT, streaming=True, context=ctx)

        ctx.write_event_to_stream(ProgressEvent(progress="Retrieving"))
        retriever = index.as_chat_engine(
            # llm=llm,
            chat_mode=ChatMode.SIMPLE,
            memory=memory,
            streaming=True,
            system_prompt=(
                "You are a chatbot, that does normal interactions, if the conversation is a greeting, respond accordingly, but if its a question, try to access the index"
            ),
            verbose=True,
            # context=ctx,
        )
        logger.info("done retrieving")
        streaming_response = retriever.stream_chat(query_str, chat_history=history)
        # streaming_response.write_response_to_history(memory=memory)

        logger.info(history)
        logger.info(memory)
        return StopEvent(result={"response": streaming_response})


class MiragWorkflow(Workflow):
    @step(pass_context=True)
    async def prepare_mirag(self, ctx: Context, ev: MiRAGQueryStartEvent) -> PrepEvent | StopEvent:
        """Prepare for retrieval."""

        llm: LLM = ev.llm
        index: VectorStoreIndex = ev.index
        searxng: SearXNGClient = ev.searxng
        query_str = ev.query_str
        history: List = ev.history
        memory: ChatMemoryBuffer = ev.memory

        await ctx.set("llm", llm)
        await ctx.set("index", index)
        await ctx.set("searxng", searxng)
        await ctx.set("query_str", query_str)
        await ctx.set("history", history)
        await ctx.set("memory", memory)

        ctx.write_event_to_stream(ProgressEvent(progress="Retrieving"))
        eval_query = await llm.acomplete(prompt=EVALUATE_QUERY.format(query_str=query_str))
        # this will respond: "question", "conversational", or "neither"
        if eval_query.text == "question":
            return PrepEvent()
        else:
            retriever = index.as_chat_engine(
                llm=llm,
                chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
                memory=memory,
                streaming=True,
                system_prompt=(
                    "You are a chatbot, that does normal interactions, if the conversation is a greeting, respond accordingly, but if its a question, try to access the index"
                ),
                context=ctx,
            )

            streaming_response = retriever.stream_chat(query_str, chat_history=history)
            # streaming_response.write_response_to_history(memory=memory)
            logger.info(history)
            logger.info(memory)

            return StopEvent(result={"response": streaming_response})

    @step(pass_context=True, num_workers=8)
    async def retrieve(self, ctx: Context, ev: PrepEvent) -> RetrieveEvent:
        """Retrieve the relevant nodes for the query."""
        query_str = await ctx.get("query_str")
        index: VectorStoreIndex = await ctx.get("index")
        searxng = await ctx.get("searxng")

        if not (index or searxng):
            raise ValueError("Index and searxng must be constructed. Run with 'documents' and 'searxng' params first.")

        ctx.write_event_to_stream(ProgressEvent(progress="Retrieving"))

        retriever = index.as_retriever(similarity_top_k=4)
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

        ctx.write_event_to_stream(ProgressEvent(progress="Checking relevance"))

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

        ctx.write_event_to_stream(ProgressEvent(progress="Extracting"))

        relevant_texts = [retrieved_nodes[i].text for i, result in enumerate(relevancy_results) if result == "yes"]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)

    @step(pass_context=True)
    async def transform_query_pipeline(self, ctx: Context, ev: TextExtractEvent) -> QueryEvent:
        """Search the transformed query with SearXNG."""
        relevant_text = ev.relevant_text
        query_str = await ctx.get("query_str")
        relevancy_score = await ctx.get("relevancy_score")

        ctx.write_event_to_stream(ProgressEvent(progress="Transforming queries"))
        if relevancy_score < 0.5:
            logger.debug("LongRAG context insufficient - transforming query and using external search")
            llm: LLM = await ctx.get("llm")
            transformed_query_str = await llm.acomplete(
                prompt=DEFAULT_TRANSFORM_QUERY_TEMPLATE.format(query_str=query_str)
            )
            logger.debug(f"Transformed query string: {transformed_query_str}")

            searxng: SearXNGClient = await ctx.get("searxng")
            searxng_results = await searxng.get_content_for_llm(query=transformed_query_str.text, max_results=10)

            search_text = "\n".join([result.content for result in searxng_results])
        else:
            logger.debug("LongRAG context fully relevant - no external search needed")
            search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step(pass_context=True)
    async def query_mirag(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        llm: LLM = await ctx.get("llm")
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        query_str = await ctx.get("query_str")
        relevancy_score = await ctx.get("relevancy_score")
        history = await ctx.get("history")

        ctx.write_event_to_stream(ProgressEvent(progress="Querying"))

        if relevancy_score > 0.5:
            status = "correct"
        elif relevancy_score == 0.0:
            status = "incorrect"
        else:
            status = "ambiguous"

        context_with_attribution = f"[Document]: {relevant_text}\n\n[Web Search]: {search_text}"

        memory = await ctx.get("memory")
        index = VectorStoreIndex(nodes=[TextNode(text=relevant_text), TextNode(text=search_text)])

        engine = index.as_chat_engine(
            llm=llm,
            chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
            memory=memory,
            streaming=True,
            system_prompt=(
                "You are a chatbot, that does normal interactions, if the conversation is a greeting, respond accordingly, but if its a question, try to access the index"
            ),
            context=ctx,
        )
        logger.info("done retrieving")
        streaming_response = engine.stream_chat(query_str, chat_history=history)

        return StopEvent(
            result={
                "response": streaming_response,
                "status": status,
            }
        )
