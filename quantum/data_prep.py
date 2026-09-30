"""
quantum/data_prep.py — Data preparation for the quantum comparison
==============================================================================

SOURCE: datasets/processed/employee_burnout_clean.csv  (22,750 rows)

    Employee ID           dropped — identifier, no predictive content
    Date of Joining       dropped — see note below
    Gender                -> binary
    Company Type          -> binary
    WFH Setup Available   -> binary
    Designation           numeric 0-5
    Resource Allocation   numeric 1-10
    Mental Fatigue Score  numeric 0-10
    Burn Rate             TARGET, continuous 0-1

WHY THIS DATASET AND NOT `outputs/burnout_scores.csv`
    `Burn Rate` comes from human survey responses. It is not derivable from the
    input features by any rule, so a classifier that predicts it has genuinely
    learnt something.

    `risk_level` in Person 2's output is the opposite: it is *defined* by a
    rule over its own inputs. Training on it would produce a model that
    relearns the rule at ~100% accuracy and proves nothing. Detailed in the
    package docstring.

SIX FEATURES, SIX QUBITS, NO PCA
    After encoding there are exactly six features, so each maps to one qubit
    under angle encoding with no dimensionality reduction at all.

    This is worth more than convenience. Most quantum-ML papers on tabular data
    must PCA their features down to fit a qubit budget, which discards variance
    and gives the classical baseline an unfair advantage (it sees the full
    feature set) or requires crippling it to match. Here both model families
    see *identical* inputs, so any accuracy difference is attributable to the
    classifier rather than to preprocessing. PCA support is retained for
    smaller qubit budgets, and is reported when used.

DATE OF JOINING
    Tenure is plausibly predictive and could be derived from this column. It is
    dropped rather than used because doing so keeps the feature count at
    exactly six — see above. Adding it is a documented extension, not an
    oversight.

SCALING TO [0, pi]
    Angle encoding writes each feature as a rotation angle. Beyond 2*pi the
    rotation wraps, so two very different feature values would map to the same
    quantum state. Scaling to [0, pi] keeps the mapping injective and uses half
    the Bloch sphere, which is standard practice.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from quantum import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DATASET = REPO_ROOT / "datasets" / "processed" / "employee_burnout_clean.csv"

#: Binary features, mapped to 0/1.
BINARY_FEATURES = {
    "Gender": {"Male": 0, "Female": 1},
    "Company Type": {"Service": 0, "Product": 1},
    "WFH Setup Available": {"No": 0, "Yes": 1},
}

#: Numeric features used as-is.
NUMERIC_FEATURES = ["Designation", "Resource Allocation", "Mental Fatigue Score"]

TARGET_COLUMN = "Burn Rate"

#: Binary threshold on Burn Rate. 0.5 sits just above the mean (0.452) and
#: median (0.45), giving a ~40/60 split — imbalanced enough to be realistic,
#: balanced enough that accuracy remains interpretable alongside F1.
BINARY_THRESHOLD = 0.5

#: Tertile boundaries for the three-class variant, chosen to mirror the
#: project's own Low / Medium / High language.
TERTILE_BOUNDS = (0.35, 0.60)


@dataclass
class PreparedData:
    """Train/test arrays plus everything needed to describe the experiment."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    feature_names: List[str]
    class_names: List[str]
    n_features: int
    n_qubits: int

    pca_applied: bool = False
    pca_explained_variance: Optional[float] = None
    scaler: object = None
    pca: object = None

    source_rows: int = 0
    sampled_rows: int = 0
    target_mode: str = "binary"
    warnings: List[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def describe(self) -> Dict[str, object]:
        """Experiment metadata, reproduced in the benchmark output."""
        return {
            "source_rows": self.source_rows,
            "sampled_rows": self.sampled_rows,
            "train_size": int(len(self.x_train)),
            "test_size": int(len(self.x_test)),
            "n_features": self.n_features,
            "n_qubits": self.n_qubits,
            "features": self.feature_names,
            "target_mode": self.target_mode,
            "classes": self.class_names,
            "class_balance_train": {
                name: int((self.y_train == index).sum())
                for index, name in enumerate(self.class_names)
            },
            "class_balance_test": {
                name: int((self.y_test == index).sum())
                for index, name in enumerate(self.class_names)
            },
            "pca_applied": self.pca_applied,
            "pca_explained_variance": self.pca_explained_variance,
            "warnings": self.warnings,
        }


# ==============================================================================
# Loading
# ==============================================================================
def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Employee Burnout dataset."""
    path = Path(path or DEFAULT_DATASET)
    if not path.is_file():
        raise FileNotFoundError(
            f"The Employee Burnout dataset was not found at:\n    {path}\n\n"
            f"This is Person 1's processed output. Without it the quantum "
            f"models have no labelled data to train on.\n"
            f"Place `employee_burnout_clean.csv` in datasets/processed/."
        )

    frame = pd.read_csv(path)
    logger.info("Loaded %s: %d rows, %d columns", path.name, *frame.shape)
    return frame


def encode_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Encode categoricals and select the modelling features."""
    encoded = pd.DataFrame(index=frame.index)
    names: List[str] = []

    for column, mapping in BINARY_FEATURES.items():
        if column not in frame.columns:
            continue
        encoded[column] = frame[column].map(mapping)
        names.append(column)

    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            continue
        encoded[column] = pd.to_numeric(frame[column], errors="coerce")
        names.append(column)

    return encoded, names


def make_labels(
    burn_rate: pd.Series, mode: str = "binary"
) -> Tuple[np.ndarray, List[str]]:
    """Convert the continuous Burn Rate into class labels.

    Args:
        burn_rate: The raw 0-1 target.
        mode: ``"binary"`` (at risk vs not) or ``"tertile"`` (Low/Medium/High).

    Binary is the default because variational quantum classifiers are far
    better established, faster to train and more stable for two classes.
    Multi-class VQC requires either one-vs-rest ensembles or more measured
    qubits, both of which add cost and variance without adding insight here.
    """
    if mode == "tertile":
        low, high = TERTILE_BOUNDS
        labels = np.digitize(burn_rate.to_numpy(), bins=[low, high])
        return labels.astype(int), ["Low", "Medium", "High"]

    labels = (burn_rate.to_numpy() >= BINARY_THRESHOLD).astype(int)
    return labels, ["Not at risk", "At risk"]


# ==============================================================================
# Preparation
# ==============================================================================
def prepare(
    path: Optional[Path] = None,
    n_qubits: int = 6,
    max_samples: Optional[int] = 1500,
    test_size: float = 0.25,
    target_mode: str = "binary",
    random_state: int = 42,
) -> PreparedData:
    """Produce train/test arrays ready for both classical and quantum models.

    Args:
        path: Dataset path. Defaults to the Employee Burnout CSV.
        n_qubits: Qubit budget. PCA is applied only if features exceed it.
        max_samples: Subsample size. Quantum kernels need O(n^2) circuit
            evaluations, so 22,750 rows is computationally infeasible on a
            simulator. ``None`` uses everything (classical models only).
        test_size: Test fraction.
        target_mode: ``"binary"`` or ``"tertile"``.
        random_state: Fixed for reproducibility across every model.

    Returns:
        A :class:`PreparedData` carrying identical inputs for all models.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    warnings: List[str] = []
    frame = load_raw(path)
    source_rows = len(frame)

    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is missing from the dataset. Columns present: "
            f"{list(frame.columns)}"
        )

    features, names = encode_features(frame)
    features[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")

    before = len(features)
    features = features.dropna()
    if len(features) < before:
        warnings.append(
            f"Dropped {before - len(features)} row(s) with missing values "
            f"({(before - len(features)) / before * 100:.1f}%)."
        )

    labels, class_names = make_labels(features[TARGET_COLUMN], target_mode)
    matrix = features[names].to_numpy(dtype=float)

    # --- subsample BEFORE splitting, stratified -----------------------------
    # Quantum kernel methods build an n x n Gram matrix, so cost is quadratic.
    # Stratified sampling preserves the class balance of the full dataset, so
    # the subsample remains representative.
    sampled_rows = len(matrix)
    if max_samples and len(matrix) > max_samples:
        from sklearn.model_selection import train_test_split as _split

        matrix, _, labels, _ = _split(
            matrix,
            labels,
            train_size=max_samples,
            stratify=labels,
            random_state=random_state,
        )
        sampled_rows = len(matrix)
        warnings.append(
            f"Subsampled {max_samples} of {source_rows} rows. Quantum kernel "
            f"methods require O(n^2) circuit evaluations, so the full dataset "
            f"is not tractable on a simulator. Sampling is stratified, so the "
            f"class balance is preserved."
        )

    # --- scale to [0, pi] for angle encoding --------------------------------
    scaler = MinMaxScaler(feature_range=(0.0, float(np.pi)))
    matrix = scaler.fit_transform(matrix)

    # --- PCA only if the qubit budget demands it ----------------------------
    pca = None
    pca_variance = None
    pca_applied = False

    if matrix.shape[1] > n_qubits:
        pca = PCA(n_components=n_qubits, random_state=random_state)
        matrix = pca.fit_transform(matrix)
        pca_variance = float(pca.explained_variance_ratio_.sum())
        pca_applied = True
        names = [f"PC{i + 1}" for i in range(n_qubits)]
        warnings.append(
            f"PCA reduced {len(NUMERIC_FEATURES) + len(BINARY_FEATURES)} "
            f"features to {n_qubits} components, retaining "
            f"{pca_variance:.1%} of variance."
        )
        # Re-scale: PCA output is centred and can be negative, which would
        # produce rotation angles outside the intended range.
        matrix = MinMaxScaler(feature_range=(0.0, float(np.pi))).fit_transform(matrix)
    elif matrix.shape[1] < n_qubits:
        warnings.append(
            f"{matrix.shape[1]} features for {n_qubits} qubits; the surplus "
            f"qubits carry no data and add depth without information."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    logger.info(
        "Prepared %d train / %d test, %d features, %d qubits, classes=%s",
        len(x_train), len(x_test), matrix.shape[1], n_qubits, class_names,
    )

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=names,
        class_names=class_names,
        n_features=int(matrix.shape[1]),
        n_qubits=n_qubits,
        pca_applied=pca_applied,
        pca_explained_variance=pca_variance,
        scaler=scaler,
        pca=pca,
        source_rows=source_rows,
        sampled_rows=sampled_rows,
        target_mode=target_mode,
        warnings=warnings,
    )


__all__ = [
    "PreparedData",
    "prepare",
    "load_raw",
    "encode_features",
    "make_labels",
    "DEFAULT_DATASET",
    "TARGET_COLUMN",
    "BINARY_THRESHOLD",
]


if __name__ == "__main__":  # pragma: no cover
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    data = prepare()
    print(json.dumps(data.describe(), indent=2))
