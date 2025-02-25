import os
import logging
from datasets import Dataset, IterableDataset, load_dataset
from dotenv import load_dotenv
from icecream import ic
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers import StringIterableReader
from tqdm.asyncio import tqdm

from crag.utils import CorrectiveRAGWorkflow
import time

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting the workflow")

    logger.info("Loading dataset from TIGER-Lab/LongRAG...")
    test_data = load_dataset(
        "TIGER-Lab/LongRAG", "nq", split="subset_100[:5]", num_proc=8
    )

    logger.info("Dataset loaded with %d records", len(test_data))
    # ic(type(test_data.select_columns(["context", "context_titles"])))
    # return

    # documents = SimpleDirectoryReader("./data").load_data()
    documents = StringIterableReader().load_data(texts=test_data[0]["context"])

    ic(type(documents))

    logger.info("Initializing the CorrectiveRAGWorkflow...")
    workflow = CorrectiveRAGWorkflow(timeout=60)

    ic(type(test_data[0]["context"]))
    logger.info("Generating index using the dataset...")
    index = await workflow.run(documents=documents)

    all_results = []
    logger.info("Starting to process queries...")

    logger.info("Processing query_id: %s", test_data[0]["query_id"])
    response = await workflow.run(
        query_str=test_data[0]["query"],
        index=index,
        tavily_ai_apikey=os.getenv("TAVILY_API_KEY"),
    )
    logger.info("Completed processing query_id: %s", test_data[0]["query_id"])

    all_results.append(
        {
            "query_id": test_data[0]["query_id"],
            "query": test_data[0]["query"],
            "result": response,
        }
    )

    # all_results.append(
    #     {"query_id": test_data["query_id"], "query": query_str, "result": response}
    # )

    output_path = "crag_out.txt"
    logger.info("Writing results to %s", output_path)
    with open(output_path, "w") as f:
        f.write(str(all_results))
    logger.info("Results successfully written to %s", output_path)

    if all_results:
        ic(all_results[0])
        logger.info("First query result: %s", all_results[0])


if __name__ == "__main__":
    import asyncio

    logger.info("Starting main execution.")
    asyncio.run(main())
    logger.info("CRAG execution finished.")
