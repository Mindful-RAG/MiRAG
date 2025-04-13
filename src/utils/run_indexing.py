import logging
import sys
import chromadb
from chromadb.config import Settings as ChromaSettings
from datasets import load_dataset
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.openai import OpenAI
from tqdm import tqdm

from longrag.utils import get_grouped_docs, split_doc
from mirag.constants import DEFAULT_CHUNK_SIZE, DEFAULT_SMALL_CHUNK_SIZE

import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-large-en-v1.5", embed_batch_size=64, cache_folder="./.embeddings", device="cuda"
# )
# Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)


async def run(persist_path="nq_corpus", collection_name="nq_corpus", batch_size=500):
    try:
        # load existing index
        db = chromadb.PersistentClient(path=persist_path, settings=ChromaSettings(anonymized_telemetry=False))
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        docstore = SimpleDocumentStore.from_persist_dir(persist_path)
        index_store = SimpleIndexStore.from_persist_dir(persist_path)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=docstore, index_store=index_store
        )
        index = load_index_from_storage(storage_context, store_nodes_override=True)
    except:
        # new index
        db = chromadb.PersistentClient(path=persist_path, settings=ChromaSettings(anonymized_telemetry=False))
        chroma_collection = db.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=SimpleDocumentStore(), index_store=SimpleIndexStore()
        )

        index = VectorStoreIndex([], storage_context=storage_context, store_nodes_override=True)

    ds = load_dataset("TIGER-Lab/LongRAG", "nq", split="full", num_proc=8)

    all_nodes = []
    for i, item in enumerate(tqdm(ds, desc="Indexing"), start=1):
        try:
            # converts dir to Document
            docs = Document(id_=f"nq_{item['query_id']}", text=item["context"])

            # longrag process
            nodes = split_doc(DEFAULT_CHUNK_SIZE, [docs])  # split documents into chunks of chunk_size
            grouped_nodes = get_grouped_docs(
                nodes
            )  # get list of nodes after grouping (groups are combined into one node), these are long retrieval units

            # split large retrieval units into smaller nodes
            small_nodes = split_doc(DEFAULT_SMALL_CHUNK_SIZE, grouped_nodes)

            all_nodes.extend(small_nodes)

            if len(all_nodes) >= batch_size:
                index.insert_nodes(all_nodes)
                index.storage_context.persist(persist_dir=persist_path)
                all_nodes = []  # Reset batch

        except Exception as e:
            with open(".iprogress", "w") as check:
                check.write(f"stopped: {str(i)}")
            print(f"Error processing document {i}: {e}")
            return

    # Insert any remaining nodes
    if all_nodes:
        index.insert_nodes(all_nodes)
        index.storage_context.persist(persist_dir=persist_path)

    logging.info("Indexing completed")


if __name__ == "__main__":
    import asyncio
    import time

    logging.info("Starting indexing process")
    start = time.time()
    asyncio.run(run())
    runtime = time.time() - start
    logging.info(f"Indexing time: {runtime:.2f} seconds")
