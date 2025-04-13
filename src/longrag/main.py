import chromadb
from datasets import load_dataset
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.settings import Settings
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
import logging
import sys
import os
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.storage.docstore.dynamodb import DynamoDBDocumentStore
from llama_index.storage.index_store.dynamodb import DynamoDBIndexStore
from llama_index.vector_stores.dynamodb import DynamoDBVectorStore

# from llama_index.llms.gemini import Gemini
from longrag.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LongRAGWorkflow,
)

import nest_asyncio


load_dotenv()


nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


async def run():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        embed_batch_size=64,
        cache_folder="./.embeddings",
        device="cuda",
    )
    wf = LongRAGWorkflow(timeout=None, verbose=True)
    # llm = Gemini(model="models/gemini-2.0-flash")
    llm = OpenAI("gpt-4o-mini")
    data_dir = "data"
    hf_dataset = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_100")

    # draw_all_possible_flows(wf, filename="longrag_workflow.html")

    # client = chromadb.HttpClient(host="localhost", port=8100)
    # client.delete_collection("longrag_store")
    # collection = client.get_or_create_collection("longrag_store")
    # vector_store = ChromaVectorStore(chroma_collection=collection)
    # db = chromadb.PersistentClient(path="./chroma_db")
    # chroma_collection = db.get_or_create_collection("longrag_collection")

    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # docstore = SimpleDocumentStore.from_persist_dir(persist_dir="./chroma_db")
    # index_store = SimpleIndexStore.from_persist_dir(persist_dir="./chroma_db")
    # storage_context = StorageContext.from_defaults(
    # docstore=docstore,
    # docstore=SimpleDocumentStore(),
    # vector_store=vector_store,
    # index_store=index_store,
    # index_store=SimpleIndexStore(),
    # persist_dir="./long_rag_index",
    # )
    # storage_context = StorageContext.from_defaults(
    #     persist_dir="./chroma_db",
    # )

    # index_kwargs = {
    #     "use_async": True,
    #     "storage_context": storage_context,
    #     "show_progress": True,
    # "store_nodes_override": True,
    # }
    index_kwargs = {"use_async": True, "show_progress": True}
    # initialize the workflow
    # index = load_index_from_storage(
    #     storage_context,
    #     # store_nodes_override=True,
    # )
    # logger.debug(index)
    result = await wf.run(
        data_dir=data_dir,
        # data_dir=hf_dataset["context"],
        # data_dir=[],
        # index=index,
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
        index_kwargs=index_kwargs,
    )
    # result["index"].storage_context.persist(persist_dir="./chroma_db")
    # result["index"].storage_context.from_defaults(
    #     docstore=docstore,
    #     vector_store=vector_store,
    #     index_store=index_store,
    # )
    # result["index"].storage_context.persist("./long_rag_index")
    query_engine = result["query_engine"]

    # results = {}
    # for query_str in hf_dataset["query"]:
    #     print(f"Running query: {query_str}")
    #     res = await wf.run(
    #         query_str=query_str,
    #         query_eng=query_engine,
    #     )
    #     results[query_str] = str(res)
    #     print(f"Query result '{query_str}': {res}")

    # run a query
    # Iterate over each query in the dataset and run it.
    # query_str = "what is the meaning of PULSE Foundation?"
    query_str = "what is the meaning of life"
    # logger.debug(result["index"].as_retriever(similarity_top_k=DEFAULT_TOP_K).retrieve(query_str))

    res = await wf.run(
        # retriever=result["retriever"],
        query_str=query_str,
        query_eng=query_engine,
    )
    logger.info(str(res))
    # Save results to a file
    # with open("longrag_res.json", "w") as f:
    #     f.write(str(results))


if __name__ == "__main__":
    import asyncio

    logger.info("Starting main execution.")
    asyncio.run(run())
    logger.info("LongRAG execution finished.")
