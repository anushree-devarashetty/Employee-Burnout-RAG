"""
quantum — Quantum machine learning comparative study (Person 5)
==============================================================================

RESEARCH QUESTION
    Can a quantum classifier predict employee burnout risk as accurately as a
    classical one, on the same features and the same split?

    Not "is quantum better". At this scale, on a simulator, it will not be.
    The contribution is a controlled, honest comparison — which is a real
    result, and the one the literature supports.

WHY THIS SITS BESIDE PERSON 2'S SCORE, NOT INSIDE IT
    The project's core claim is explainability: a transparent rule an HR
    manager can read. A variational quantum circuit is the opposite of
    interpretable. So the quantum model is a SECOND OPINION shown alongside the
    rule-based score, never a replacement for it.

THE LABEL TRAP THIS MODULE DELIBERATELY AVOIDS
    The obvious target is `risk_level` from `outputs/burnout_scores.csv`. That
    would be worthless: the column is produced by a deterministic rule
    (`Negative +2, sadness +1`, thresholds at 2 and 5), so any classifier
    trained on it merely relearns the rule, scores ~100%, and demonstrates
    nothing.

    The target used here is `Burn Rate` from `employee_burnout_clean.csv` —
    a real value from human survey responses, not derivable from the input
    features by any rule. That makes the comparison meaningful.

MODULES
    data_prep.py           load, encode, scale, split
    circuits.py            feature maps + ansätze, shared by both frameworks
    classical_baseline.py  LogisticRegression, SVM-RBF, RandomForest
    pennylane_models.py    VQC (PyTorch autograd) + quantum-kernel SVM
    qiskit_models.py       VQC + QSVC (Qiskit Machine Learning)
    benchmark.py           runs everything, writes the comparison table

RUN
    python quantum/benchmark.py --qubits 6 --samples 1500
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__version__ = "1.0.0"

__all__ = ["REPO_ROOT", "__version__"]
