import json

from loguru import logger


class ResultHandler:
    @staticmethod
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
                        "rouge_scores": results["rouge_scores"],
                    }
                )
            )

    @staticmethod
    def write_final_summary(
        args,
        dataset_size,
        successful_items,
        context_sizes,
        exact_match,
        substring_match,
        correct,
        ambiguous,
        incorrect,
        failed_items,
        retried_items,
        rouge_scores,
    ):
        """Write final summary statistics to file"""
        logger.success(f"Successfully processed: {successful_items}/{dataset_size} items")
        logger.success(f"Failed items: {len(failed_items)}/{dataset_size}")
        logger.success(f"Retried successfully: {len(retried_items)}")

        if context_sizes:
            logger.success(f"Context size: {sum(context_sizes) / len(context_sizes)}")

        # Only calculate metrics on successfully processed items
        if successful_items > 0:
            logger.success(f"Exact Match: {exact_match / successful_items}")
            logger.success(f"Substring Match: {substring_match / successful_items}")
            logger.success(f"Correct Match: {correct / successful_items}")
            logger.success(f"Ambiguous Match: {ambiguous / successful_items}")
            logger.success(f"Incorrect Match: {incorrect / successful_items}")
            if args.lfqa:
                logger.success(f"Rouge(precision): {rouge_scores['rouge1']['precision']}")
                logger.success(f"Rouge(recall): {rouge_scores['rouge1']['recall']}")
                logger.success(f"Rouge(fmeasure): {rouge_scores['rouge1']['fmeasure']}")
        else:
            logger.warning("No items were successfully processed")

        # Configuration information
        config_info = {"llm": args.llm, "embed_model": args.embed_model, "data_name": args.data_name}

        rouge  = {}
        if args.lfqa:
            rouge = rouge_scores

        # Write summary to file
        with open(f"summary_{args.output_file}", "w") as f:
            f.write(
                json.dumps(
                    {
                        "configuration": config_info,
                        "dataset_size": dataset_size,
                        "processed_items": successful_items,
                        "context_size": sum(context_sizes) / len(context_sizes) if context_sizes else None,
                        "exact_match": exact_match / successful_items if successful_items > 0 else 0,
                        "substring_match": substring_match / successful_items if successful_items > 0 else 0,
                        "correct_match": correct / successful_items if successful_items > 0 else 0,
                        "ambiguous_match": ambiguous / successful_items if successful_items > 0 else 0,
                        "incorrect_match": incorrect / successful_items if successful_items > 0 else 0,
                        "failed_items": [item["query_id"] for item in failed_items],
                        "retried_successfully": retried_items,
                        "completion_percentage": (successful_items / dataset_size) * 100,
                        "rouge_scores": rouge,
                    }
                )
            )
