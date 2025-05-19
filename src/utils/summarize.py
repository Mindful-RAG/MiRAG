import json

from mirag.benchmarks import rouge_metric


def summarize_jsonl(input_path, output_path=None):
    # Read all items
    with open(input_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

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

    summary = {
        "dataset_size": len(results),
        "processed_items": processed,
        "error_items": errors,
        "exact_match": exact_match / processed if processed > 0 else 0,
        "substring_match": substring_match / processed if processed > 0 else 0,
        "correct_match": correct / processed if processed > 0 else 0,
        "ambiguous_match": ambiguous / processed if processed > 0 else 0,
        "incorrect_match": incorrect / processed if processed > 0 else 0,
        "failed_item_ids": [r.get("id") for r in results if r.get("status") == "error"],
        "completion_percentage": processed / len(results) * 100 if len(results) > 0 else 0,
    }

    # Optionally, add ROUGE scores here if you want (see below)
    if args.rouge:
        rouge_scores, _ = rouge_metric(results, prediction_key="long_answer", answer_key="answer")
        summary["rouge_scores"] = rouge_scores
    # Write summary
    if output_path is None:
        output_path = f"summary_{input_path.split('/')[-1]}"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Output")
    parser.add_argument(
        "--results-path",
        type=str,
        help="Path to results file (jsonl)",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        help="Path to output summary file (jsonl)",
    )

    parser.add_argument(
        "--rouge",
        action="store_true",
        help="rouge metric",
    )
    args = parser.parse_args()

    summarize_jsonl(args.results_path, args.summary_path)
