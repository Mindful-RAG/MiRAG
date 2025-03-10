import nest_asyncio
from datasets.load import load_dataset
from dotenv import load_dotenv
from icecream import ic

from llama_index.llms.openai import OpenAI
# from llama_index.llms.gemini import Gemini

from longrag.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LongRAGWorkflow,
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
)
from utils.logger import logger

load_dotenv()

nest_asyncio.apply()


async def run():
    wf = LongRAGWorkflow(timeout=60)
    # llm = Gemini(model="models/gemini-2.0-flash")
    llm = OpenAI("gpt-4o-mini")
    data_dir = "data"
    hf_dataset = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_100[:5]", trust_remote_code=True)

    # draw_all_possible_flows(wf, filename="longrag_workflow.html")

    # initialize the workflow
    result = await wf.run(
        # data_dir=data_dir,
        data_dir=hf_dataset["context"],
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    )

    ic(result)
    query_engine = result["query_engine"]

    results = {}
    for query_str in hf_dataset["query"]:
        print(f"Running query: {query_str}")
        res = await wf.run(
            query_str=query_str,
            query_eng=query_engine,
        )
        results[query_str] = str(res)
        print(f"Query result '{query_str}': {res}")

    # run a query
    # Iterate over each query in the dataset and run it.
    # query_str = "What is the capital of France?"
    # res = await wf.run(
    #     query_str=query_str,
    #     query_eng=query_engine,
    # )
    # Save results to a file
    with open("longrag_res.json", "w") as f:
        f.write(str(results))


if __name__ == "__main__":
    import uvloop

    logger.info("Starting main execution.")
    uvloop.run(run())
    logger.info("LongRAG execution finished.")
