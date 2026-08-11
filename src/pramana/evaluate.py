"""RAGAS-based evaluation.

Overall accuracy can't tell you whether a wrong answer came from bad
retrieval, an over-eager prompt, or a chunking gap — three different causes
needing three different fixes. RAGAS decomposes the pipeline into dimensions
that point at a specific layer to fix. This module wraps that scoring as a
first-class part of the library rather than something bolted on separately
once a project already exists.
"""

from __future__ import annotations

from pramana.grounding import GroundedAnswerer
from pramana.types import EvalResult, TestCase


def evaluate(
    answerer: GroundedAnswerer,
    test_cases: list[TestCase],
    mlflow_uri: str | None = None,
) -> EvalResult:
    """Run `test_cases` through `answerer` and score the results with RAGAS.

    Always scores Faithfulness and Context Precision — the two dimensions
    that don't require ground-truth labels. Context Recall and Answer
    Relevancy are added automatically for any test case that carries
    `relevant_chunk_ids` / a non-empty `expected_answer`, matching the
    book's own progression (Book 1 ships two dimensions because that's what
    an unlabeled test set supports; the other two require labels you
    provide).

    Requires the `eval` extra: `pip install pramana-rag[eval]`.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "evaluate() requires the 'ragas' and 'datasets' packages. "
            "Install with: pip install pramana-rag[eval]"
        ) from exc

    rows = []
    has_ground_truth = False
    for case in test_cases:
        evidence = answerer.retriever.retrieve(case.query)
        contexts = [item.chunk.text for item in evidence] if evidence else [""]
        generated = answerer.answer(case.query)
        if case.relevant_chunk_ids or case.expected_answer:
            has_ground_truth = True
        rows.append(
            {
                "question": case.query,
                "answer": generated,
                "contexts": contexts,
                "ground_truth": case.expected_answer,
            }
        )

    dataset = Dataset.from_list(rows)

    metrics = [faithfulness, context_precision]
    if has_ground_truth:
        metrics += [context_recall, answer_relevancy]

    result = ragas_evaluate(dataset, metrics=metrics)
    scores = dict(result)

    per_query = [dict(row) for row in result.scores] if hasattr(result, "scores") else []

    eval_result = EvalResult(
        faithfulness=scores.get("faithfulness"),
        context_precision=scores.get("context_precision"),
        context_recall=scores.get("context_recall"),
        answer_relevancy=scores.get("answer_relevancy"),
        per_query=per_query,
    )

    if mlflow_uri:
        _log_to_mlflow(mlflow_uri, eval_result, len(test_cases))

    return eval_result


def _log_to_mlflow(uri: str, result: EvalResult, total_queries: int) -> None:
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("pramana-eval")
    with mlflow.start_run():
        mlflow.log_param("total_queries", total_queries)
        metrics = {
            "faithfulness": result.faithfulness,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "answer_relevancy": result.answer_relevancy,
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
