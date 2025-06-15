# This service takes in the user uploaded data to be indexed and will be available to /longrag and /mirag endpoints


from mirag.index_management import IndexManager
from api.config import env_vars
from api.utils.observability import logger


class IndexCorpus:
    def __init__(self, wf, embed_model, llm):
        self.wf = wf
        self.llm = llm
        self.embed_model = embed_model

        self._set_index_manager()

    def _set_index_manager(self):
        try:
            self.index_manager = IndexManager(env_vars.PERSIST_PATH, self.wf, self.llm)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize IndexManager: {str(e)}")

    async def index_corpus(self, args, document):
        try:
            logger.debug(args)
            index = await self.index_manager.load_or_create_index(
                args=args, dataset=document, embed_model=self.embed_model
            )
            if index is None:
                raise ValueError("Failed to create or load index. Please check your configuration and dataset.")
        except Exception as e:
            raise ValueError(f"Error during indexing: {str(e)}")

        return index
