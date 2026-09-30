"""
quantum/classical_baseline.py — The comparison the quantum models must beat
==============================================================================

WHY THE BASELINE MATTERS MORE THAN THE QUANTUM MODELS

    A quantum classifier reporting 88% accuracy sounds impressive until you
    learn that logistic regression gets 89% on the same split in 4
    milliseconds. Without a baseline, a quantum result is unfalsifiable — and
    an examiner's first question will be "compared to what?".

    Three baselines are used, chosen to bracket the problem:

      DummyClassifier    predicts the majority class. The floor. Any model
                         that fails to beat this has learnt nothing, and on an
                         imbalanced split it can look deceptively strong on
                         accuracy alone — which is exactly why F1 is reported.

      LogisticRegression a linear decision boundary. If this matches the
                         quantum model, the problem is linearly separable and
                         no expressive classifier of any kind was required.

      SVM (RBF kernel)   the direct classical counterpart to a quantum kernel
                         method. QSVC replaces the RBF kernel with a quantum
                         fidelity kernel and changes nothing else, so this is
                         the single most informative comparison in the study.

      RandomForest       a strong non-linear ensemble. Represents "what you
                         would actually deploy".

FAIRNESS CONSTRAINTS
    Every model receives the identical scaled, split arrays from `data_prep`.
    Same random seed, same subsample, same PCA (or none). Nothing is tuned per
    model. Any accuracy difference is therefore attributable to the classifier,
    not to preprocessing or data volume.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np

from quantum.data_prep import PreparedData

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Metrics for one trained model.

    Shared by the classical and quantum implementations so the benchmark table
    is directly comparable row to row.
    """

    name: str
    family: str                      # "classical" | "quantum"
    framework: str                   # "scikit-learn" | "pennylane" | "qiskit"

    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: Optional[float] = None

    train_seconds: float = 0.0
    predict_seconds: float = 0.0

    n_parameters: Optional[int] = None
    confusion: Optional[List[List[int]]] = None
    notes: List[str] = field(default_factory=list)
    failed: bool = False
    error: Optional[str] = None

    def row(self) -> Dict[str, Any]:
        """Flat dict for the results CSV."""
        return {
            "model": self.name,
            "family": self.family,
            "framework": self.framework,
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else "",
            "train_seconds": round(self.train_seconds, 3),
            "predict_seconds": round(self.predict_seconds, 3),
            "n_parameters": self.n_parameters if self.n_parameters is not None else "",
            "failed": self.failed,
            "error": self.error or "",
        }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the metric set used for every model.

    `average="weighted"` for F1/precision/recall because the classes are
    imbalanced (~40/60 binary). Macro would over-weight the minority class;
    weighted reflects real-world performance while still — unlike raw accuracy
    — penalising a model that ignores one class entirely.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "roc_auc": None,
    }

    # ROC-AUC is only defined for binary problems with a score, and requires
    # both classes present in y_true.
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass

    return metrics


def _ranking_scores(model, x, n_classes: int):
    """Continuous scores for ROC-AUC, without requiring probabilities.

    ROC-AUC only needs a RANKING, not calibrated probabilities. Preferring
    `decision_function` avoids `SVC(probability=True)` — deprecated in
    scikit-learn 1.9, removed in 1.11 — and skips its expensive internal
    cross-validated calibration. Falls back to `predict_proba` for models
    that expose only that (Random Forest), and to None when neither exists
    (DummyClassifier), in which case AUC is simply not reported.
    """
    if n_classes != 2:
        return None
    for attribute, extract in (
        ("decision_function", lambda v: v),
        ("predict_proba", lambda v: v[:, 1]),
    ):
        method = getattr(model, attribute, None)
        if callable(method):
            try:
                return extract(method(x))
            except Exception:
                continue
    return None


def run_classical_baselines(data: PreparedData) -> List[ModelResult]:
    """Train and evaluate every classical baseline on the prepared split."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = [
        (
            "Majority baseline",
            DummyClassifier(strategy="most_frequent"),
            "Predicts the most common class. The floor any model must clear.",
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=42),
            "Linear boundary. Matching this means the task needed no "
            "expressive model.",
        ),
        (
            "SVM (RBF kernel)",
            # NOT probability=True. That is deprecated in scikit-learn 1.9 and
            # removed in 1.11, and it was always the wrong tool here: it runs
            # an internal 5-fold Platt calibration purely to produce
            # probabilities we only need for ROC-AUC. `decision_function`
            # gives the signed margin, which ranks identically and therefore
            # yields the same AUC, at a fraction of the training cost.
            SVC(kernel="rbf", random_state=42),
            "The direct classical counterpart to the quantum kernel method.",
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            "Strong non-linear ensemble; what you would actually deploy.",
        ),
    ]

    results: List[ModelResult] = []

    for name, model, note in models:
        result = ModelResult(name=name, family="classical", framework="scikit-learn")
        result.notes.append(note)

        try:
            started = time.perf_counter()
            model.fit(data.x_train, data.y_train)
            result.train_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = model.predict(data.x_test)
            result.predict_seconds = time.perf_counter() - started

            scores = _ranking_scores(model, data.x_test, data.n_classes)

            metrics = evaluate(data.y_test, predictions, scores)
            result.accuracy = metrics["accuracy"]
            result.f1 = metrics["f1"]
            result.precision = metrics["precision"]
            result.recall = metrics["recall"]
            result.roc_auc = metrics["roc_auc"]
            result.confusion = metrics["confusion"]

            if hasattr(model, "coef_"):
                result.n_parameters = int(np.size(model.coef_) + np.size(model.intercept_))

            logger.info(
                "%-22s acc=%.4f f1=%.4f  train=%.3fs",
                name, result.accuracy, result.f1, result.train_seconds,
            )

        except Exception as exc:
            result.failed = True
            result.error = str(exc)
            logger.error("%s failed: %s", name, exc)

        results.append(result)

    return results


__all__ = ["ModelResult", "evaluate", "run_classical_baselines"]


if __name__ == "__main__":  # pragma: no cover
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    from quantum.data_prep import prepare

    data = prepare()
    for result in run_classical_baselines(data):
        print(json.dumps(result.row(), indent=2))
