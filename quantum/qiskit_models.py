"""
quantum/qiskit_models.py — VQC and QSVC in Qiskit Machine Learning
==============================================================================

WHY IMPLEMENT THE SAME MODELS TWICE

    Not redundancy — validation.

    A single implementation of a quantum classifier gives you a number with no
    way to tell whether it reflects the algorithm or a bug in your circuit. Two
    independent implementations of the SAME specification, in two libraries
    written by different teams, agreeing to within noise, is real evidence that
    the result is the algorithm's rather than an artefact.

    If PennyLane reports 87% and Qiskit reports 62% on identical data, one of
    them is wrong and the study has found that before an examiner does.

    Both build from `circuits.CircuitSpec`, so the structure cannot drift.

DIFFERENCES THAT ARE NOT BUGS
    - Qiskit's `ZZFeatureMap` uses a slightly different phase convention to the
      hand-built PennyLane map. Kernel values differ marginally; classification
      accuracy should not.
    - Qiskit's `VQC` minimises cross-entropy over a one-hot output by default,
      where the PennyLane implementation minimises square loss on +/-1 labels.
      Both are standard; convergence paths differ, final accuracy should not.
    - Qiskit's COBYLA is gradient-free; PennyLane uses Adam with analytic
      gradients. COBYLA is often more robust on small parameter counts, and is
      the usual choice in Qiskit tutorials.

    These are documented rather than eliminated, because forcing the two
    libraries into byte-identical behaviour would mean abandoning each one's
    idiomatic API — and the comparison is meant to reflect what a practitioner
    would actually write.

VERSION
    Written against qiskit 2.x and qiskit-machine-learning 0.9.1, which
    migrated to Qiskit 2.0 and V2 primitives.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np

from quantum.circuits import CircuitSpec, DEFAULT_SPEC
from quantum.classical_baseline import ModelResult, evaluate
from quantum.data_prep import PreparedData

logger = logging.getLogger(__name__)


def _require_qiskit():
    try:
        import qiskit  # noqa: F401
        import qiskit_machine_learning  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Qiskit is not installed.\n"
            "  Fix: pip install -r quantum/requirements.txt"
        ) from exc


# ==============================================================================
# Circuit construction
# ==============================================================================
def build_feature_map(spec: CircuitSpec):
    """Build the Qiskit feature map matching `spec`.

    ``angle`` uses ZFeatureMap with one repetition — a Hadamard plus one
    phase rotation per qubit, the closest Qiskit equivalent to the
    single-rotation angle encoding used in PennyLane.

    ``zz`` uses ZZFeatureMap, which adds entangled rotations on feature
    products. This is the map Havlicek et al. showed induces a classically
    hard-to-estimate kernel.
    """
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap

    if spec.feature_map == "zz":
        return ZZFeatureMap(
            feature_dimension=spec.n_qubits, reps=1, entanglement="linear"
        )
    return ZFeatureMap(feature_dimension=spec.n_qubits, reps=1)


def build_ansatz(spec: CircuitSpec):
    """Build the trainable circuit matching `spec`.

    `RealAmplitudes` is Qiskit's standard hardware-efficient ansatz: layers of
    RY rotations separated by CNOT entanglement — structurally the same family
    as the PennyLane ansatz, with RY rotations and a CNOT ring.
    """
    from qiskit.circuit.library import RealAmplitudes

    entanglement = "circular" if spec.entanglement == "ring" else spec.entanglement
    return RealAmplitudes(
        num_qubits=spec.n_qubits, reps=spec.n_layers, entanglement=entanglement
    )


# ==============================================================================
# Runner
# ==============================================================================
def run_qiskit_models(
    data: PreparedData,
    spec: Optional[CircuitSpec] = None,
    max_iterations: int = 60,
    kernel_max_train: int = 300,
) -> List[ModelResult]:
    """Train and evaluate the Qiskit VQC and QSVC."""
    _require_qiskit()
    spec = spec or CircuitSpec(n_qubits=data.n_qubits)
    results: List[ModelResult] = []

    # --- VQC ------------------------------------------------------------------
    vqc_result = ModelResult(
        name="VQC (variational)", family="quantum", framework="qiskit"
    )
    try:
        from qiskit_algorithms.optimizers import COBYLA
    except ImportError:
        try:
            from qiskit_machine_learning.optimizers import COBYLA
        except ImportError:
            COBYLA = None

    try:
        if COBYLA is None:
            raise ImportError(
                "No COBYLA optimizer found in qiskit_algorithms or "
                "qiskit_machine_learning.optimizers."
            )

        from qiskit_machine_learning.algorithms.classifiers import VQC

        feature_map = build_feature_map(spec)
        ansatz = build_ansatz(spec)

        vqc_result.n_parameters = ansatz.num_parameters
        vqc_result.notes.append(
            f"{spec.n_qubits} qubits, RealAmplitudes reps={spec.n_layers}, "
            f"{ansatz.num_parameters} parameters, COBYLA "
            f"maxiter={max_iterations}."
        )
        vqc_result.notes.append(
            "COBYLA is gradient-free, unlike the PennyLane implementation's "
            "Adam with analytic gradients. Convergence paths differ; final "
            "accuracy should not."
        )

        logger.info("Training Qiskit VQC (%d parameters)...", ansatz.num_parameters)
        model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=max_iterations),
        )

        started = time.perf_counter()
        model.fit(data.x_train, data.y_train)
        vqc_result.train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = np.asarray(model.predict(data.x_test)).ravel().astype(int)
        vqc_result.predict_seconds = time.perf_counter() - started

        metrics = evaluate(data.y_test, predictions)
        vqc_result.accuracy = metrics["accuracy"]
        vqc_result.f1 = metrics["f1"]
        vqc_result.precision = metrics["precision"]
        vqc_result.recall = metrics["recall"]
        vqc_result.confusion = metrics["confusion"]

    except Exception as exc:
        vqc_result.failed = True
        vqc_result.error = str(exc)
        logger.error("Qiskit VQC failed: %s", exc)

    results.append(vqc_result)

    # --- QSVC -----------------------------------------------------------------
    qsvc_result = ModelResult(
        name="Quantum kernel SVM", family="quantum", framework="qiskit"
    )
    try:
        from qiskit_machine_learning.algorithms import QSVC
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        n_train = min(kernel_max_train, len(data.x_train))
        x_train = data.x_train[:n_train]
        y_train = data.y_train[:n_train]

        qsvc_result.notes.append(
            f"FidelityQuantumKernel on {n_train} of {len(data.x_train)} rows "
            f"({n_train ** 2:,} kernel entries)."
        )

        logger.info("Computing Qiskit quantum kernel on %d rows...", n_train)
        kernel = FidelityQuantumKernel(feature_map=build_feature_map(spec))
        model = QSVC(quantum_kernel=kernel)

        started = time.perf_counter()
        model.fit(x_train, y_train)
        qsvc_result.train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = np.asarray(model.predict(data.x_test)).ravel().astype(int)
        qsvc_result.predict_seconds = time.perf_counter() - started

        metrics = evaluate(data.y_test, predictions)
        qsvc_result.accuracy = metrics["accuracy"]
        qsvc_result.f1 = metrics["f1"]
        qsvc_result.precision = metrics["precision"]
        qsvc_result.recall = metrics["recall"]
        qsvc_result.confusion = metrics["confusion"]

    except Exception as exc:
        qsvc_result.failed = True
        qsvc_result.error = str(exc)
        logger.error("Qiskit QSVC failed: %s", exc)

    results.append(qsvc_result)
    return results


__all__ = ["run_qiskit_models", "build_feature_map", "build_ansatz"]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    from quantum.data_prep import prepare

    data = prepare(max_samples=300)
    for result in run_qiskit_models(data, max_iterations=30, kernel_max_train=120):
        print(result.row())
