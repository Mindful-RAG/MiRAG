import os
import traceback

from loguru import logger

from mirag.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
)


class IndexManager:
    def __init__(self, persist_path, wf, llm):
        self.persist_path = persist_path
        self.wf = wf
        self.llm = llm

    async def load_or_create_index(self, args, dataset=None):
        """Load an existing index or create a new one based on arguments"""
        index = None

        # Try to load index from disk if requested
        if args.load_index and os.path.exists(self.persist_path):
            logger.info(f"Loading index from {self.persist_path}")
            try:
                from llama_index.core import load_index_from_storage
                from llama_index.core.storage import StorageContext

                # Load the index from disk
                storage_context = StorageContext.from_defaults(persist_dir=self.persist_path)
                loaded_index = load_index_from_storage(storage_context)

                # Run a minimal workflow step to get the index in the right format
                # We just need to convert the loaded index into the format expected by the workflow
                index = await self.wf.run(
                    dataset=[],  # Empty dataset since we're loading the index
                    llm=self.llm,
                    index=loaded_index,  # Pass the loaded index
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    similarity_top_k=DEFAULT_TOP_K,
                    small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
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
            index = await self.wf.run(
                dataset=dataset,
                llm=self.llm,
                chunk_size=DEFAULT_CHUNK_SIZE,
                similarity_top_k=DEFAULT_TOP_K,
                small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
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
