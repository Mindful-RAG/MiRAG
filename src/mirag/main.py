import argparse
import json
import os
import time
import traceback

import tiktoken
from datasets import load_dataset

# import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from loguru import logger
from tqdm.asyncio import tqdm

from mirag.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SMALL_CHUNK_SIZE,
    DEFAULT_TOP_K,
)
from mirag.metrics import has_correct_answer, single_ans_em
from mirag.workflows import MindfulRAGWorkflow

load_dotenv()

# nest_asyncio.apply()


def parse_arguments():
    parser = argparse.ArgumentParser(description="MiRAG: Mindful RAG Workflow")

    parser.add_argument("--llm", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument(
        "--embed_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="embeddings model to use",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="mirag_output.jsonl",
        help="Output of the workflow",
    )

    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=2,
        help="Number of retry attempts for failed queries",
    )
    parser.add_argument(
        "--continue-from-file",
        action="store_true",
        help="Continue processing from a previous output file",
    )
    parser.add_argument(
        "--process-errors-only",
        action="store_true",
        help="Process only the error entries in the output file",
    )
    parser.add_argument(
        "--persist-index",
        action="store_true",
        help="Persist the index to disk",
    )
    parser.add_argument(
        "--persist-path",
        type=str,
        default="./persisted_index",
        help="Path to persist the index",
    )
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="Load index from disk instead of creating a new one",
    )

    return parser.parse_args()


async def process_item(wf, item, index, llm, tavily_api_key):
    """Process a single dataset item with error handling"""
    query, answers = item["query"], item["answer"]
    context_titles, context = item["context_titles"], item["context"]
    enc = tiktoken.get_encoding("cl100k_base")
    context_size = len(enc.encode(context))

    res = await wf.run(
        query_str=query,
        context_titles=context_titles,
        llm=llm,
        index=index["index"],
        # tavily_api_key=tavily_api_key,
    )

    is_exact_match = single_ans_em(res["short_answer"], answers)
    is_substring_match = has_correct_answer(res["long_answer"], answers)

    output = {
        "id": item["query_id"],
        "query": query,
        "answer": answers,
        "long_answer": res["long_answer"],
        "short_answer": res["short_answer"],
        "is_exact_match": is_exact_match,
        "is_substring_match": is_substring_match,
        "status": res["status"],  # correct|ambiguous|incorrect
    }

    return output, context_size


def load_previous_results(output_file):
    """Load results from a previous run"""
    results = []
    error_items = []

    if not os.path.exists(output_file):
        logger.warning(f"Output file {output_file} does not exist. Starting fresh.")
        return results, error_items

    try:
        with open(output_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                result = json.loads(line)
                results.append(result)

                if result.get("status") == "error":
                    error_items.append(result)

        logger.info(f"Loaded {len(results)} results from {output_file}")
        logger.info(f"Found {len(error_items)} error entries to process")

    except Exception as e:
        logger.error(f"Error loading previous results: {str(e)}")
        logger.error(traceback.format_exc())

    return results, error_items


def get_item_from_dataset(dataset, item_id):
    """Find an item in the dataset by its ID"""
    for item in dataset:
        if item["query_id"] == item_id:
            return item
    return None


async def continue_from_previous_file(args, dataset, wf, index, llm, tavily_api_key):
    """Continue processing from a previous output file by filling in error entries"""
    results, error_items = load_previous_results(args.output_file)

    if not error_items:
        logger.info("No error entries found in the output file.")
        return

    logger.info(f"Attempting to process {len(error_items)} error entries")

    # Statistics
    substring_match, exact_match = 0, 0
    correct, ambiguous, incorrect = 0, 0, 0
    processed = 0
    context_sizes = []
    still_failed = []

    # Process each error item
    for error_item in tqdm(error_items, desc="Processing errors"):
        item_id = error_item["id"]
        dataset_item = get_item_from_dataset(dataset, item_id)

        if not dataset_item:
            logger.error(f"Could not find item {item_id} in dataset")
            still_failed.append(error_item)
            continue

        try:
            output, context_size = await process_item(wf, dataset_item, index, llm, tavily_api_key)
            context_sizes.append(context_size)

            if output["status"] == "correct":
                correct += 1
            elif output["status"] == "ambiguous":
                ambiguous += 1
            elif output["status"] == "incorrect":
                incorrect += 1

            processed += 1
            exact_match += output["is_exact_match"]
            substring_match += output["is_substring_match"]

            # Update the result in the main results list
            for i, res in enumerate(results):
                if res.get("id") == item_id:
                    results[i] = output
                    break

        except Exception as e:
            error_message = f"Error reprocessing item {item_id}: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            still_failed.append(error_item)

    # Write updated results back to file
    with open(args.output_file, "w") as f:
        for result in results:
            json_string = json.dumps(result)
            f.write(f"{json_string}\n")

    # Report statistics
    logger.info(f"Successfully reprocessed: {processed}/{len(error_items)} error items")
    logger.info(f"Items still failing: {len(still_failed)}/{len(error_items)}")

    if processed > 0:
        logger.info(f"Exact Match: {exact_match / processed}")
        logger.info(f"Substring Match: {substring_match / processed}")
        logger.info(f"Correct Match: {correct / processed}")
        logger.info(f"Ambiguous Match: {ambiguous / processed}")
        logger.info(f"Incorrect Match: {incorrect / processed}")

    # Update the summary file
    update_summary_file(args.output_file, results)


def update_summary_file(output_file, results):
    """Update the summary file based on final results"""
    summary_file = f"summary_{output_file}"

    # Calculate overall statistics
    substring_match, exact_match = 0, 0
    correct, ambiguous, incorrect, errors = 0, 0, 0, 0

    for result in results:
        if result.get("status") == "error":
            errors += 1
        else:
            if result.get("is_exact_match", False):
                exact_match += 1
            if result.get("is_substring_match", False):
                substring_match += 1

            if result.get("status") == "correct":
                correct += 1
            elif result.get("status") == "ambiguous":
                ambiguous += 1
            elif result.get("status") == "incorrect":
                incorrect += 1

    processed = len(results) - errors

    # Write updated summary
    with open(summary_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "dataset_size": len(results),
                    "processed_items": processed,
                    "error_items": errors,
                    "exact_match": exact_match / processed if processed > 0 else 0,
                    "substring_match": substring_match / processed if processed > 0 else 0,
                    "correct_match": correct / processed if processed > 0 else 0,
                    "ambiguous_match": ambiguous / processed if processed > 0 else 0,
                    "incorrect_match": incorrect / processed if processed > 0 else 0,
                    "failed_item_ids": [r["id"] for r in results if r.get("status") == "error"],
                    "completion_percentage": processed / len(results) * 100,
                }
            )
        )


async def main():
    args = parse_arguments()

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

    wf = MindfulRAGWorkflow(timeout=60)
    # draw_all_possible_flows(wf, "mirag_workflow.html")

    logger.info("loading dataset")
    dataset = load_dataset(
        "TIGER-LAB/LongRAG",
        "nq",
        split="subset_100[:1]",
        # split="subset_1000[:5]",
        trust_remote_code=True,
    )

    # Create a list of strings that combines titles and context
    combined_texts = []
    for item in dataset:
        # Combine title and content in each string
        text = f"Title: {item['context_titles']}\n\nContent: {item['context']}"
        combined_texts.append(text)

    logger.info("loading index")

    index = None

    # Try to load index from disk if requested
    if args.load_index and os.path.exists(args.persist_path):
        logger.info(f"Loading index from {args.persist_path}")
        try:
            from llama_index.core import load_index_from_storage
            from llama_index.core.storage import StorageContext

            # Load the index from disk
            storage_context = StorageContext.from_defaults(persist_dir=args.persist_path)
            loaded_index = load_index_from_storage(storage_context)

            # Run a minimal workflow step to get the index in the right format
            # We just need to convert the loaded index into the format expected by the workflow
            index = await wf.run(
                dataset=[],  # Empty dataset since we're loading the index
                llm=llm,
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

    # Create index if it wasn't loaded
    if index is None:
        logger.info("Creating new index")
        index = await wf.run(
            dataset=combined_texts,
            llm=llm,
            chunk_size=DEFAULT_CHUNK_SIZE,
            similarity_top_k=DEFAULT_TOP_K,
            small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
        )

        # Persist the index if requested
        if args.persist_index:
            logger.info(f"Persisting index to {args.persist_path}")
            # Create directory if it doesn't exist
            os.makedirs(args.persist_path, exist_ok=True)

            try:
                # Save the index to disk
                index["index"].storage_context.persist(persist_dir=args.persist_path)
                logger.info("Successfully persisted index to disk")
            except Exception as e:
                logger.error(f"Error persisting index: {str(e)}")
                logger.error(traceback.format_exc())

    # Check if we're continuing from a previous file
    if args.continue_from_file or args.process_errors_only:
        await continue_from_previous_file(args, dataset, wf, index, llm, os.getenv("TAVILY_API_KEY"))
        return

    # index = await wf.run(
    #     dataset=combined_texts,
    #     llm=llm,
    #     chunk_size=DEFAULT_CHUNK_SIZE,
    #     similarity_top_k=DEFAULT_TOP_K,
    #     small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    # )

    # # Check if we're continuing from a previous file
    # if args.continue_from_file or args.process_errors_only:
    #     await continue_from_previous_file(
    #         args, dataset, wf, index, llm, os.getenv("TAVILY_API_KEY")
    #     )
    #     return

    logger.info("running query")

    substring_match, exact_match = 0, 0
    correct, ambiguous, incorrect = 0, 0, 0
    tt = 0
    context_sizes = []
    enc = tiktoken.get_encoding("cl100k_base")
    dataset_size = len(dataset)
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    output_file = open(args.output_file, "w")
    failed_items = []
    results = []

    for i, item in enumerate(tqdm(dataset, desc="Querying")):
        try:
            output, context_size = await process_item(wf, item, index, llm, tavily_api_key)

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

            # Write error to output file to maintain ordering
            error_output = {
                "id": item["query_id"],
                "query": item["query"],
                "error": error_message,
                "status": "error",
            }
            json_string = json.dumps(error_output)
            output_file.write(f"{json_string}\n")
            results.append(error_output)

    # Second pass - retry failed items
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
                    output, context_size = await process_item(wf, item, index, llm, tavily_api_key)
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

                    # Find and replace the error entry with successful result
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
    with open(args.output_file, "w") as f:
        for result in results:
            json_string = json.dumps(result)
            f.write(f"{json_string}\n")

    # Report final statistics
    logger.info(f"Successfully processed: {tt}/{dataset_size} items")
    logger.info(f"Failed items: {len(failed_items)}/{dataset_size}")
    logger.info(f"Retried successfully: {len(retried_items)}")

    if context_sizes:
        logger.info(f"Context size: {sum(context_sizes) / len(context_sizes)}")

    # Only calculate metrics on successfully processed items
    successful_items = tt
    if successful_items > 0:
        logger.info(f"Exact Match: {exact_match / successful_items}")
        logger.info(f"Substring Match: {substring_match / successful_items}")
        logger.info(f"Correct Match: {correct / successful_items}")
        logger.info(f"Ambiguous Match: {ambiguous / successful_items}")
        logger.info(f"Incorrect Match: {incorrect / successful_items}")
    else:
        logger.warning("No items were successfully processed")

    # Write summary to file
    with open(f"summary_{args.output_file}", "w") as f:
        f.write(
            json.dumps(
                {
                    "dataset_size": dataset_size,
                    "processed_items": tt,
                    "context_size": sum(context_sizes) / len(context_sizes) if context_sizes else None,
                    "exact_match": exact_match / successful_items if successful_items > 0 else 0,
                    "substring_match": substring_match / successful_items if successful_items > 0 else 0,
                    "correct_match": correct / successful_items if successful_items > 0 else 0,
                    "ambiguous_match": ambiguous / successful_items if successful_items > 0 else 0,
                    "incorrect_match": incorrect / successful_items if successful_items > 0 else 0,
                    "failed_items": [item["query_id"] for item in failed_items],
                    "retried_successfully": retried_items,
                    "completion_percentage": (tt / dataset_size) * 100,
                }
            )
        )


def run():
    import uvloop

    logger.info("Starting main execution.")
    start = time.time()
    uvloop.run(main())
    runtime = time.time() - start
    logger.info(f"MiRAG execution finished in {runtime:.2f} seconds.")


if __name__ == "__main__":
    run()
