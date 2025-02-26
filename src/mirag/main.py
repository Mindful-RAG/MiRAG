import os
import json
from datasets import load_dataset
from llama_index.llms.openai import OpenAI

# import nest_asyncio
from dotenv import load_dotenv
from icecream import ic
from tqdm.asyncio import tqdm

from mirag.metrics import has_correct_answer, single_ans_em
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

    substring_match, exact_match, retrieval = 0, 0, 0
    tt = 0
    context_sizes = []

    output_file = open("mirag_output.json", "w")

    results = {}
    for item in tqdm(dataset, desc="Querying"):
        query, answers = item["query"], item["answer"]
        context_titles, context = item["context_titles"], item["context"]
        res = await wf.run(
            query_str=query,
            index=index["index"],
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )

        is_exact_match = single_ans_em(res["short_answer"], answers)
        is_substring_match = has_correct_answer(res["long_answer"], answers)
        # assuming its nq

        # TODO: add appropriate query including crag (correct|ambiguous|incorrect)
        output = {
            "id": item["query_id"],
            "query": query,
            "answer": answers,
            "long_answer": res["long_answer"],
            "short_answer": res["short_answer"],
            "is_exact_match": is_exact_match,
            "is_substring_match": is_substring_match,
        }

        tt += 1
        exact_match += is_exact_match
        substring_match += is_substring_match
        if tt % 10 == 0:
            logger.info(f"Substring match: {substring_match / tt}")
            logger.info(f"Exact match: {exact_match / tt}")
        json_string = json.dumps(output)
        output_file.write(f"{json_string},\n")

    logger.info(f"Retrieval accuracy: {retrieval / len(dataset)}")
    logger.info(f"Exact Match: {exact_match / len(dataset)}")


def run():
    import uvloop

    logger.info("Starting main execution.")
    uvloop.run(main())
    logger.info("MiRAG execution finished.")


if __name__ == "__main__":
    run()
