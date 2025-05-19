import json
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

from chromadb.config import Settings as ChromaSettings
from tqdm import tqdm

# from llama_index.llms.gemini import Gemini
from longrag.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LongRAGWorkflow,
)

import nest_asyncio

from mirag.benchmarks import rouge_metric


load_dotenv()


nest_asyncio.apply()

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

logger.disable(name="longrag.utils")


async def run():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        embed_batch_size=64,
        cache_folder="./.embeddings",
        device="cuda",
    )
    persist_path = "./nq_corpus_full"
    collection_name = "nq_corpus"
    wf = LongRAGWorkflow(timeout=None)
    # llm = Gemini(model="models/gemini-2.0-flash")
    llm = OpenAI("gpt-4o-mini")
    # dataset = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_1000[:500]")

    logger.info("Starting LongRAG workflow")
    eli5 = load_dataset("sentence-transformers/eli5", split="train").select(range(500))

    eli5_shape = eli5.rename_columns({"question": "query"}).add_column(
        new_fingerprint="add_id", name="query_id", column=[f"eli5_{i}" for i in range(len(eli5))]
    )

    dataset = eli5_shape

    # draw_all_possible_flows(wf, filename="longrag_workflow.html")

    db = chromadb.PersistentClient(path=persist_path, settings=ChromaSettings(anonymized_telemetry=False))
    chroma_collection = db.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    docstore = SimpleDocumentStore.from_persist_dir(persist_path)
    index_store = SimpleIndexStore.from_persist_dir(persist_path)
    # logger.debug(len(index_store.index_structs()))
    # logger.debug(list(docstore.docs))

    # storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
        index_store=index_store,
        # persist_dir=self.persist_path
    )

    loaded_index = load_index_from_storage(storage_context, store_nodes_override=True)

    # result = await wf.run(
    #     # data_dir=data_dir,
    #     # data_dir=hf_dataset["context"],
    #     data_dir=[],
    #     index=loaded_index,
    #     llm=llm,
    #     chunk_size=DEFAULT_CHUNK_SIZE,
    #     similarity_top_k=DEFAULT_TOP_K,
    #     small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    # )

    results = []
    for i, item in enumerate(tqdm(dataset, desc="Querying")):
        query, answers = item["query"], item["answer"]
        res = await wf.run(
            llm=llm,
            index=loaded_index,
            query_str=query,
        )
        output = {"id": item["query_id"], "query": query, "answer": answers, "long_answer": res["long_answer"]}
        results.append(output)

    rouge_scores, results = rouge_metric(results, prediction_key="long_answer", answer_key="answer")

    with open("longrag_lfqa_500.jsonl", "w") as f:
        for result in results:
            json_string = json.dumps(result)
            f.write(f"{json_string}\n")
    # Save results to a file
    # with open("longrag_res.json", "w") as f:
    #     f.write(str(results))

    substring_match, exact_match = 0, 0
    correct, ambiguous, incorrect, errors = 0, 0, 0, 0

    for result in results:
        if result.get("status") == "error":
            errors += 1
        else:
            if result.get("is_exact_match", False):
                exact_match += 1
            if result.get("is_substring_match", False):
                substring_match += 1

            if result.get("status") == "correct":
                correct += 1
            elif result.get("status") == "ambiguous":
                ambiguous += 1
            elif result.get("status") == "incorrect":
                incorrect += 1

    processed = len(results) - errors

    summary = {
        "dataset_size": len(results),
        "processed_items": processed,
        "error_items": errors,
        # "exact_match": exact_match / processed if processed > 0 else 0,
        # "substring_match": substring_match / processed if processed > 0 else 0,
        # "correct_match": correct / processed if processed > 0 else 0,
        # "ambiguous_match": ambiguous / processed if processed > 0 else 0,
        # "incorrect_match": incorrect / processed if processed > 0 else 0,
        "failed_item_ids": [r.get("id") for r in results if r.get("status") == "error"],
        "completion_percentage": processed / len(results) * 100 if len(results) > 0 else 0,
    }

    summary["rouge_scores"] = rouge_scores["rouge1"]
    # Write summary
    output_path = "summary_lfqa_500.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path}")
    logger.success(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import asyncio

    logger.info("Starting main execution.")
    asyncio.run(run())
    logger.info("LongRAG execution finished.")
