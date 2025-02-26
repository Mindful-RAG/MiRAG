import os
import json
from datasets import load_dataset
from llama_index.llms.openai import OpenAI

# import nest_asyncio
from dotenv import load_dotenv
from icecream import ic

from utils.logger import logger

from mirag.workflows import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
    MindfulRAGWorkflow,
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
)

load_dotenv()

# nest_asyncio.apply()


async def main():
    wf = MindfulRAGWorkflow(timeout=60)
    # draw_all_possible_flows(wf, "mirag_workflow.html")

    logger.info("loading dataset")
    llm = OpenAI(model="gpt-4o-mini")
    dataset = load_dataset(
        "TIGER-LAB/LongRAG",
        "nq",
        split="subset_100[:5]",
        trust_remote_code=True,
    )

    logger.info("loading index")
    index = await wf.run(
        dataset=dataset["context"],
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    )

    logger.info("running query")

    results = {}
    for query_str in dataset["query"]:
        res = await wf.run(
            query_str=query_str,
            index=index["index"],
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        results[query_str] = str(res)

    logger.info("saving results")
    with open("mirag_output.txt", "w") as f:
        f.write(json.dumps(results))

    # logger.info("iello")


def run():
    import uvloop

    logger.info("Starting main execution.")
    uvloop.run(main())
    logger.info("MiRAG execution finished.")


if __name__ == "__main__":
    run()
