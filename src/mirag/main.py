import os
import json
import time
import traceback

import nest_asyncio
from datasets import load_dataset
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from loguru import logger
from tqdm.asyncio import tqdm

# from api.main import run_api
from mirag.cli import CLI
from mirag.data_processing import DataProcessor
from mirag.index_management import IndexManager
from mirag.result_handling import ResultHandler
from mirag.workflows import MindfulRAGWorkflow
from utils.async_utils import asyncio_run
from utils.searxng import SearXNGClient
from mirag.config import env_vars

nest_asyncio.apply()

load_dotenv()
logger.disable(name="mirag.workflows")


async def main():
    args = CLI.parse_arguments()
    if args.debug:
        logger.enable("mirag.workflows")

    searxng = SearXNGClient(instance_url=env_vars.SEARXNG_URL)
    await searxng._test_connection()

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embed_model,
        embed_batch_size=64,
        cache_folder="./.embeddings",
        device="cuda",
    )

    # Configure LLM based on argument
    if "gpt" in args.llm:
        llm = OpenAI(model=args.llm, temperature=0)
        Settings.llm = llm
    elif "gemini" in args.llm:
        llm = Gemini(model=f"models/{args.llm}")
        Settings.llm = llm

    wf = MindfulRAGWorkflow(timeout=None, verbose=True)
    logger.info("loading dataset")
    dataset = load_dataset("TIGER-LAB/LongRAG", args.data_name, split=args.split, trust_remote_code=True, num_proc=8)

    logger.info("loading index")
    index_manager = IndexManager(args.persist_path, wf, llm)
    index = await index_manager.load_or_create_index(args, dataset)

    # Initialize data processor
    data_processor = DataProcessor(wf, index, llm, searxng, args)

    # Check if we're continuing from a previous file
    if args.continue_from_file or args.process_errors_only:
        await data_processor.continue_from_previous_file(args, dataset)
        return

    logger.info("running query")
    substring_match, exact_match = 0, 0
    correct, ambiguous, incorrect = 0, 0, 0
    tt = 0
    context_sizes = []
    dataset_size = len(dataset)
    output_file = open(args.output_file, "w")
    failed_items = []
    results = []

    for i, item in enumerate(tqdm(dataset, desc="Querying")):
        try:
            output, context_size = await data_processor.process_item(item)
            context_sizes.append(context_size)

            if output["status"] == "correct":
                correct += 1
            elif output["status"] == "ambiguous":
                ambiguous += 1
            elif output["status"] == "incorrect":
                incorrect += 1

            tt += 1
            exact_match += output["is_exact_match"]
            substring_match += output["is_substring_match"]

            if tt % 10 == 0:
                logger.info(f"Substring match: {substring_match / tt}")
                logger.info(f"Exact match: {exact_match / tt}")

            json_string = json.dumps(output)
            output_file.write(f"{json_string}\n")
            results.append(output)

        except Exception as e:
            error_message = f"Error processing item {item['query_id']}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            failed_items.append(item)

            error_output = {
                "id": item["query_id"],
                "query": item["query"],
                "error": error_message,
                "status": "error",
            }
            json_string = json.dumps(error_output)
            output_file.write(f"{json_string}\n")
            results.append(error_output)

    # Retry failed items
    retry_count = 0
    retried_items = []

    if failed_items:
        logger.info(f"Retrying {len(failed_items)} failed queries")

        for attempt in range(args.retry_attempts):
            if not failed_items:
                break

            retry_count += 1
            logger.info(f"Retry attempt {retry_count}")

            items_to_retry = failed_items.copy()
            failed_items = []

            for item in tqdm(items_to_retry, desc=f"Retry #{retry_count}"):
                try:
                    output, context_size = await data_processor.process_item(item)
                    context_sizes.append(context_size)

                    if output["status"] == "correct":
                        correct += 1
                    elif output["status"] == "ambiguous":
                        ambiguous += 1
                    elif output["status"] == "incorrect":
                        incorrect += 1

                    tt += 1
                    exact_match += output["is_exact_match"]
                    substring_match += output["is_substring_match"]

                    for i, res in enumerate(results):
                        if res.get("id") == item["query_id"] and res.get("status") == "error":
                            results[i] = output
                            break

                    retried_items.append(item["query_id"])

                except Exception as e:
                    error_message = f"Error in retry #{retry_count} for item {item['query_id']}: {str(e)}"
                    logger.error(error_message)
                    logger.error(traceback.format_exc())
                    failed_items.append(item)

    output_file.close()
    await searxng.close()
    with open(args.output_file, "w") as f:
        for result in results:
            json_string = json.dumps(result)
            f.write(f"{json_string}\n")

    ResultHandler.write_final_summary(
        args,
        dataset_size,
        tt,
        context_sizes,
        exact_match,
        substring_match,
        correct,
        ambiguous,
        incorrect,
        failed_items,
        retried_items,
    )


def run():
    """Entry point for the MiRAG application"""

    args = CLI.parse_arguments()

    logger.info("Starting MiRAG execution.")
    start = time.time()
    asyncio_run(main())
    runtime = time.time() - start
    logger.info(f"MiRAG execution finished in {runtime:.2f} seconds.")


if __name__ == "__main__":
    run()
