import os
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import traceback
from chromadb.config import Settings
import chromadb
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.storage import StorageContext
from loguru import logger
from dotenv import load_dotenv
from pinecone.control.types.create_index_for_model_embed import Metric

from mirag.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
)

load_dotenv()


class IndexManager:
    def __init__(self, persist_path, wf, llm):
        self.persist_path = persist_path
        self.wf = wf
        self.llm = llm
        self.collection_name = "mirag-collection"
        self.db = chromadb.PersistentClient(path=self.persist_path, settings=Settings(anonymized_telemetry=False))

    async def load_or_create_index(self, args, dataset=None):
        """Load an existing index or create a new one based on arguments"""
        index = None

        # Try to load index from disk if requested
        if args.load_index and os.path.exists(self.persist_path):
            logger.info(f"Loading index from {self.persist_path}")
            try:
                from llama_index.core import load_index_from_storage

                chroma_collection = self.db.get_collection(self.collection_name)
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

                storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_path)

                loaded_index = load_index_from_storage(storage_context)

                # Load the index from disk
                # storage_context = StorageContext.from_defaults(persist_dir=self.persist_path)
                # loaded_index = load_index_from_storage(storage_context)

                # Run a minimal workflow step to get the index in the right format
                # We just need to convert the loaded index into the format expected by the workflow
                index = await self.wf.run(
                    dataset=[],  # Empty dataset since we're loading the index
                    llm=self.llm,
                    index=loaded_index,  # Pass the loaded index
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    similarity_top_k=DEFAULT_TOP_K,
                    small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
                    index_kwargs={"use_async": True},
                )
                logger.info("Successfully loaded index from disk")

            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                logger.error(traceback.format_exc())
                logger.info("Will create a new index instead")
                index = None

        # Create index if it wasn't loaded and dataset is provided
        if index is None and dataset is not None:
            logger.info("Creating new index")

            try:
                self.db.delete_collection(self.collection_name)
            except:
                pass

            chroma_collection = self.db.create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index_kwargs = {"use_async": True, "storage_context": storage_context, "show_progress": True}

            index = await self.wf.run(
                dataset=dataset,
                llm=self.llm,
                chunk_size=DEFAULT_CHUNK_SIZE,
                similarity_top_k=DEFAULT_TOP_K,
                small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
                index_kwargs=index_kwargs,
            )

            # Persist the index if requested
            if args.persist_index:
                self.persist_index(index)

        return index

    def persist_index(self, index):
        """Persist the index to disk"""
        logger.info(f"Persisting index to {self.persist_path}")
        # Create directory if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)

        try:
            # Save the index to disk
            index["index"].storage_context.persist(persist_dir=self.persist_path)
            logger.info("Successfully persisted index to disk")
        except Exception as e:
            logger.error(f"Error persisting index: {str(e)}")
            logger.error(traceback.format_exc())
