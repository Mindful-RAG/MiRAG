from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from rouge_score import rouge_scorer


class RougeMetric(BaseModel):
    rouge: Dict[str, Dict[str, float]]


def rouge_metric(
    results: List[Dict[str, Any]], prediction_key: str, answer_key: str
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    sum_scores = {}
    count = 0
    for result in results:
        scorer = rouge_scorer.RougeScorer(rouge_types=["rouge1", "rougeL"], use_stemmer=True)
        prediction = result[prediction_key]
        answers = result[answer_key]

        best_scores = {}
        for ref in answers:
            scores = scorer.score(prediction=prediction, target=ref)
            for rouge_type, score in scores.items():
                score_dict = score._asdict()
                if rouge_type not in best_scores:
                    best_scores[rouge_type] = score_dict
                else:
                    # Take max for each metric
                    for metric in score_dict:
                        best_scores[rouge_type][metric] = max(best_scores[rouge_type][metric], score_dict[metric])
        scores_dict = RougeMetric(rouge=best_scores).model_dump()["rouge"]

        result["rouge"] = scores_dict
        for rouge_type, metrics in scores_dict.items():
            if rouge_type not in sum_scores:
                sum_scores[rouge_type] = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            for metric, value in metrics.items():
                sum_scores[rouge_type][metric] += value
        count += 1

    avg_scores = {
        rouge_type: {metric: value / count for metric, value in metrics.items()}
        for rouge_type, metrics in sum_scores.items()
    }
    return avg_scores, results
