import json
import traceback

import tiktoken
from loguru import logger
from tqdm.asyncio import tqdm


class DataProcessor:
    def __init__(self, wf, index, llm, searxng, args):
        self.wf = wf
        self.index = index
        self.llm = llm
        self.searxng = searxng
        self.args = args

    async def process_item(self, item):
        """Process a single dataset item with error handling"""
        query, answers = item["query"], item["answer"]
        context_titles, context = item["context_titles"], item["context"]
        enc = tiktoken.get_encoding("cl100k_base")
        context_size = len(enc.encode(context))

        res = await self.wf.run(
            query_str=query,
            context_titles=context_titles,
            llm=self.llm,
            data_name=self.args.data_name,
            index=self.index["index"],
            searxng=self.searxng,
        )

        is_exact_match = single_ans_em(res["short_answer"], answers)
        is_substring_match = has_correct_answer(res["long_answer"], answers)

        output = {
            "id": item["query_id"],
            "query": query,
            "answer": answers,
            "long_answer": res["long_answer"].lower().strip(),
            "short_answer": res["short_answer"].lower().strip(),
            "is_exact_match": is_exact_match,
            "is_substring_match": is_substring_match,
            "status": res["status"],  # correct|ambiguous|incorrect
        }

        return output, context_size

    @staticmethod
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

    @staticmethod
    def get_item_from_dataset(dataset, item_id):
        """Find an item in the dataset by its ID"""
        for item in dataset:
            if item["query_id"] == item_id:
                return item
        return None

    async def continue_from_previous_file(self, args, dataset):
        """Continue processing from a previous output file by filling in error entries"""
        from mirag.result_handling import ResultHandler

        results, error_items = self.load_previous_results(args.output_file)

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
            dataset_item = self.get_item_from_dataset(dataset, item_id)

            if not dataset_item:
                logger.error(f"Could not find item {item_id} in dataset")
                still_failed.append(error_item)
                continue

            try:
                output, context_size = await self.process_item(dataset_item)
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
        ResultHandler.update_summary_file(args.output_file, results)


# Import functions to avoid circular imports
import os

from mirag.metrics import has_correct_answer, single_ans_em
