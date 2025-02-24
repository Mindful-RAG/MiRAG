import nest_asyncio
from datasets.load import load_dataset
from dotenv import load_dotenv
from icecream import ic

# from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

from longrag.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
    LongRAGWorkflow,
)

load_dotenv()

nest_asyncio.apply()


# from IPython.display import display, Markdown

wf = LongRAGWorkflow(timeout=60)
llm = Gemini(model="models/gemini-2.0-flash")
# llm = OpenAI("gpt-4o-mini")
data_dir = "data"
test_data = load_dataset("TIGER-Lab/LongRAG", "nq", split="subset_100")


async def run():
    # initialize the workflow
    result = await wf.run(
        data_dir=None,
        dataset=test_data,
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    )

    query_engine = result["query_engine"]

    # run a query
    # Iterate over each query in the dataset and run it.
    all_results = []
    for record in test_data:
        # for record in test_data:
        query_str = record["query"]
        res = await wf.run(
            query_str=query_str,
            query_eng=query_engine,
        )
        all_results.append(
            {
                "query_id": record["query_id"],
                "query": query_str,
                "result": res,
            }
        )

    # Save results to a file
    with open("res.txt", "w") as f:
        f.write(str(all_results))


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
