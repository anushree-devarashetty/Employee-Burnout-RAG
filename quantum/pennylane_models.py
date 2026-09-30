"""
quantum/pennylane_models.py — VQC and quantum-kernel SVM in PennyLane
==============================================================================

TWO QUANTUM APPROACHES, DELIBERATELY DIFFERENT IN KIND

1.  VARIATIONAL QUANTUM CLASSIFIER (VQC)
        A parameterised circuit trained by gradient descent. The quantum
        circuit is the model; a classical optimiser tunes its rotation angles.

            x --[feature map]--[ansatz W(θ)]--[<Z>]--> prediction
                                     ^                      |
                                     +---- Adam ------------+

        Analogous to a small neural network, with the same failure modes:
        local minima, learning-rate sensitivity, and the quantum-specific one
        — barren plateaus.

2.  QUANTUM KERNEL SVM
        No variational training at all. Encode two points, measure the overlap
        of their states, and use that fidelity as a kernel entry:

            K(xi, xj) = |<phi(xi)|phi(xj)>|^2

        Feed the Gram matrix to a classical SVM. The quantum part computes
        similarity; the SVM does the learning.

        Usually the better choice at this scale: deterministic, no training
        instability, no plateaus. The cost is O(n^2) circuit evaluations, which
        is why `data_prep` subsamples.

WHY THE PARAMETER-SHIFT RULE MATTERS
    Gradients of a quantum circuit cannot be obtained by ordinary
    backpropagation, because measuring a quantum state destroys it. The
    parameter-shift rule recovers exact analytic gradients by evaluating the
    circuit twice per parameter at shifted angles:

        d<Z>/dθ = [ f(θ + π/2) - f(θ - π/2) ] / 2

    This is why quantum training is slow: each gradient step costs 2P circuit
    evaluations for P parameters. On the default simulator PennyLane can use
    backprop through the state vector instead, which is far faster and
    mathematically equivalent — but only because we are *simulating*. On real
    hardware, parameter-shift is unavoidable, and that cost is a genuine
    finding worth reporting.
==============================================================================
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np

from quantum.circuits import CircuitSpec, DEFAULT_SPEC, parameter_shape
from quantum.classical_baseline import ModelResult, evaluate
from quantum.data_prep import PreparedData

logger = logging.getLogger(__name__)


def _require_pennylane():
    try:
        import pennylane as qml

        return qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is not installed.\n"
            "  Fix: pip install -r quantum/requirements.txt"
        ) from exc


# ==============================================================================
# Circuit construction
# ==============================================================================
def build_feature_map(qml, spec: CircuitSpec):
    """Return a function that encodes one sample into the circuit."""

    def angle_map(x):
        # x[..., wire] works for both a single sample (n_features,) and a
        # broadcast batch (batch, n_features) — the ellipsis preserves any
        # leading batch dimension.
        for wire in range(spec.n_qubits):
            qml.RY(x[..., wire], wires=wire)

    def zz_map(x):
        # Second-order (ZZ) map from Havlicek et al. 2019: Hadamards to create
        # superposition, RZ for each feature, then entangled RZ on feature
        # PRODUCTS. The product terms are what make the induced feature space
        # classically hard to compute.
        for wire in range(spec.n_qubits):
            qml.Hadamard(wires=wire)
            qml.RZ(x[..., wire], wires=wire)
        for first, second in spec.entangling_pairs():
            qml.CNOT(wires=[first, second])
            qml.RZ(x[..., first] * x[..., second], wires=second)
            qml.CNOT(wires=[first, second])

    return zz_map if spec.feature_map == "zz" else angle_map


def build_ansatz(qml, spec: CircuitSpec):
    """Return a function applying the trainable layers."""

    def ansatz(weights):
        for layer in range(spec.n_layers):
            for wire in range(spec.n_qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
                if spec.ansatz == "strong":
                    qml.RX(weights[layer, wire, 2], wires=wire)
            # Entangling ring: without this the circuit is n independent
            # single-qubit classifiers and cannot model feature interactions.
            for first, second in spec.entangling_pairs():
                qml.CNOT(wires=[first, second])

    return ansatz


# ==============================================================================
# Variational Quantum Classifier
# ==============================================================================
class PennyLaneVQC:
    """Binary variational quantum classifier."""

    def __init__(
        self,
        spec: CircuitSpec = DEFAULT_SPEC,
        epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.spec = spec
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    def _make_circuit(self):
        qml = _require_pennylane()

        # "default.qubit" is a state-vector simulator. diff_method="backprop"
        # differentiates through the simulated state, which is exact and far
        # faster than parameter-shift — valid only because we simulate.
        device = qml.device("default.qubit", wires=self.spec.n_qubits)
        feature_map = build_feature_map(qml, self.spec)
        ansatz = build_ansatz(qml, self.spec)

        @qml.qnode(device, interface="autograd", diff_method="backprop")
        def circuit(weights, x):
            # PARAMETER BROADCASTING: `x` may be a single sample of shape
            # (n_features,) OR a whole batch of shape (batch, n_features).
            # PennyLane broadcasts the rotation angles across the leading
            # dimension and evaluates the entire batch in one vectorised pass.
            #
            # This is not a micro-optimisation. Looping over samples in Python
            # and calling the QNode once each was ~30s per epoch on 1,125 rows;
            # broadcasting evaluates the same epoch in a fraction of that,
            # because the simulator applies one batched matrix operation
            # instead of n separate circuit constructions.
            feature_map(x)
            ansatz(weights)
            # Expectation of Pauli-Z on qubit 0, in [-1, 1]. Sign gives the
            # class; magnitude gives confidence.
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PennyLaneVQC":
        """Train by minimising square loss on the +/-1 encoded labels."""
        qml = _require_pennylane()
        from pennylane import numpy as pnp

        rng = np.random.default_rng(self.seed)
        circuit = self._make_circuit()
        shape = parameter_shape(self.spec)

        # Small initial weights. Large random initialisation lands the circuit
        # in a barren plateau immediately, where gradients are ~0.
        weights = pnp.array(rng.normal(0, 0.1, size=shape), requires_grad=True)
        bias = pnp.array(0.0, requires_grad=True)

        # Labels to {-1, +1} to match the range of <Z>.
        targets = pnp.array(np.where(y == 1, 1.0, -1.0), requires_grad=False)
        features = pnp.array(x, requires_grad=False)

        optimiser = qml.AdamOptimizer(stepsize=self.learning_rate)

        def cost(weights, bias, batch_x, batch_y):
            # One broadcast call for the whole batch instead of a Python loop.
            predictions = circuit(weights, batch_x) + bias
            return pnp.mean((predictions - batch_y) ** 2)

        n_samples = len(features)
        for epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, n_samples, self.batch_size):
                index = order[start : start + self.batch_size]
                batch_x, batch_y = features[index], targets[index]

                # step_and_cost returns the loss it already computed, so the
                # cost function is not evaluated a second time just to log it.
                (weights, bias, _, _), loss = optimiser.step_and_cost(
                    cost, weights, bias, batch_x, batch_y
                )
                epoch_loss += float(loss)
                batches += 1

            mean_loss = epoch_loss / max(batches, 1)
            self.loss_history.append(mean_loss)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info("  epoch %2d/%d  loss=%.4f", epoch + 1, self.epochs,
                            mean_loss)

            # Flat loss across several epochs is the signature of a barren
            # plateau. Worth surfacing rather than burning ten more minutes.
            if len(self.loss_history) > 6:
                recent = self.loss_history[-6:]
                if max(recent) - min(recent) < 1e-5:
                    logger.warning(
                        "Loss has not moved in 6 epochs — likely a barren "
                        "plateau or a learning rate that is too small."
                    )
                    break

        self.weights = np.array(weights)
        self.bias = float(bias)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() before predicting.")
        circuit = self._make_circuit()
        return np.array(circuit(self.weights, np.asarray(x))) + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.decision_function(x) > 0).astype(int)

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Map <Z> from [-1, 1] to [0, 1] for ROC-AUC."""
        return (np.clip(self.decision_function(x), -1, 1) + 1) / 2


# ==============================================================================
# Quantum kernel SVM
# ==============================================================================
class PennyLaneKernelSVM:
    """Classical SVM using a quantum fidelity kernel."""

    def __init__(self, spec: CircuitSpec = DEFAULT_SPEC, C: float = 1.0) -> None:
        self.spec = spec
        self.C = C
        self.svm = None
        self.x_train: Optional[np.ndarray] = None

    def _make_kernel_circuit(self):
        qml = _require_pennylane()

        device = qml.device("default.qubit", wires=self.spec.n_qubits)
        feature_map = build_feature_map(qml, self.spec)

        @qml.qnode(device)
        def kernel_circuit(x1, x2):
            # Encode x1, then apply the INVERSE encoding of x2. The probability
            # of measuring the all-zero state is then exactly the fidelity
            # |<phi(x1)|phi(x2)>|^2 — the kernel value.
            feature_map(x1)
            qml.adjoint(feature_map)(x2)
            return qml.probs(wires=range(self.spec.n_qubits))

        return kernel_circuit

    def _gram(self, a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """Kernel matrix between two sets of samples.

        O(len(a) x len(b)) circuit evaluations — inherently quadratic, and the
        reason `data_prep` subsamples. Measured on this dataset: 100 rows ->
        5.2s, 150 -> 10.8s, 200 -> 21.6s. Doubling the rows quadruples the
        cost, exactly as the complexity predicts.

        Two optimisations reduce the constant factor without changing that
        complexity:

        1. SYMMETRY. K(xi, xj) = K(xj, xi), and the diagonal is exactly 1
           (a state has perfect fidelity with itself). So only the upper
           triangle is computed — just under half the work.

        2. BROADCASTING. All required pairs are evaluated in batched calls
           rather than one Python-level call per pair. The simulator applies
           one vectorised operation over the batch, which is far faster than
           reconstructing the circuit tens of thousands of times.

        `chunk` bounds peak memory: each batched call holds `chunk` state
        vectors of 2^n_qubits amplitudes.
        """
        circuit = self._make_kernel_circuit()

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        symmetric = a.shape == b.shape and np.array_equal(a, b)

        matrix = np.zeros((len(a), len(b)))

        # Collect the index pairs that actually need evaluating.
        if symmetric:
            np.fill_diagonal(matrix, 1.0)   # identical states, fidelity = 1
            pairs = [(i, j) for i in range(len(a)) for j in range(i + 1, len(b))]
        else:
            pairs = [(i, j) for i in range(len(a)) for j in range(len(b))]

        if not pairs:
            return matrix

        left_index = np.fromiter((i for i, _ in pairs), dtype=int, count=len(pairs))
        right_index = np.fromiter((j for _, j in pairs), dtype=int, count=len(pairs))

        for start in range(0, len(pairs), chunk):
            end = start + chunk
            batch_left = a[left_index[start:end]]
            batch_right = b[right_index[start:end]]

            # Probability of the all-zero outcome == |<phi(x1)|phi(x2)>|^2.
            probabilities = np.asarray(circuit(batch_left, batch_right))
            values = probabilities[..., 0]

            matrix[left_index[start:end], right_index[start:end]] = values
            if symmetric:
                matrix[right_index[start:end], left_index[start:end]] = values

        return matrix

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PennyLaneKernelSVM":
        from sklearn.svm import SVC

        self.x_train = x
        gram = self._gram(x, x)
        # Precomputed quantum kernel. No probability=True: deprecated in
        # scikit-learn 1.9, and `decision_function` on the precomputed Gram
        # matrix already gives the ranking ROC-AUC needs.
        self.svm = SVC(kernel="precomputed", C=self.C, random_state=42)
        self.svm.fit(gram, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.svm is None:
            raise RuntimeError("Call fit() before predicting.")
        return self.svm.predict(self._gram(x, self.x_train))

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Signed margin from the SVM, used for ROC-AUC."""
        return self.svm.decision_function(self._gram(x, self.x_train))


# ==============================================================================
# Runner
# ==============================================================================
def run_pennylane_models(
    data: PreparedData,
    spec: Optional[CircuitSpec] = None,
    epochs: int = 30,
    kernel_max_train: int = 300,
) -> List[ModelResult]:
    """Train and evaluate both PennyLane models.

    Args:
        kernel_max_train: Cap on training rows for the kernel method. The Gram
            matrix is n^2 circuit evaluations, so 1,125 rows would be ~1.3
            million circuits — hours on a simulator. The cap is reported in
            the results so the comparison stays honest.
    """
    spec = spec or CircuitSpec(n_qubits=data.n_qubits)
    results: List[ModelResult] = []

    warning = spec.barren_plateau_warning()

    # --- VQC ------------------------------------------------------------------
    vqc_result = ModelResult(
        name="VQC (variational)", family="quantum", framework="pennylane"
    )
    vqc_result.n_parameters = spec.n_parameters
    vqc_result.notes.append(
        f"{spec.n_qubits} qubits, {spec.n_layers} layers, "
        f"{spec.feature_map} feature map, {spec.n_parameters} trainable parameters."
    )
    if warning:
        vqc_result.notes.append(warning)

    try:
        logger.info("Training PennyLane VQC (%d parameters)...", spec.n_parameters)
        model = PennyLaneVQC(spec=spec, epochs=epochs)

        started = time.perf_counter()
        model.fit(data.x_train, data.y_train)
        vqc_result.train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(data.x_test)
        vqc_result.predict_seconds = time.perf_counter() - started

        metrics = evaluate(data.y_test, predictions, model.predict_score(data.x_test))
        vqc_result.accuracy = metrics["accuracy"]
        vqc_result.f1 = metrics["f1"]
        vqc_result.precision = metrics["precision"]
        vqc_result.recall = metrics["recall"]
        vqc_result.roc_auc = metrics["roc_auc"]
        vqc_result.confusion = metrics["confusion"]
        vqc_result.notes.append(
            f"Final training loss {model.loss_history[-1]:.4f} after "
            f"{len(model.loss_history)} epochs."
        )
    except Exception as exc:
        vqc_result.failed = True
        vqc_result.error = str(exc)
        logger.error("PennyLane VQC failed: %s", exc)

    results.append(vqc_result)

    # --- Quantum kernel SVM ---------------------------------------------------
    kernel_result = ModelResult(
        name="Quantum kernel SVM", family="quantum", framework="pennylane"
    )
    try:
        n_train = min(kernel_max_train, len(data.x_train))
        x_train = data.x_train[:n_train]
        y_train = data.y_train[:n_train]

        kernel_result.notes.append(
            f"Trained on {n_train} of {len(data.x_train)} rows: the Gram matrix "
            f"costs O(n^2) circuit evaluations ({n_train ** 2:,} circuits here)."
        )

        logger.info("Computing quantum kernel on %d training rows...", n_train)
        model = PennyLaneKernelSVM(spec=spec)

        started = time.perf_counter()
        model.fit(x_train, y_train)
        kernel_result.train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        predictions = model.predict(data.x_test)
        kernel_result.predict_seconds = time.perf_counter() - started

        metrics = evaluate(data.y_test, predictions, model.predict_score(data.x_test))
        kernel_result.accuracy = metrics["accuracy"]
        kernel_result.f1 = metrics["f1"]
        kernel_result.precision = metrics["precision"]
        kernel_result.recall = metrics["recall"]
        kernel_result.roc_auc = metrics["roc_auc"]
        kernel_result.confusion = metrics["confusion"]
    except Exception as exc:
        kernel_result.failed = True
        kernel_result.error = str(exc)
        logger.error("Quantum kernel SVM failed: %s", exc)

    results.append(kernel_result)
    return results


__all__ = [
    "PennyLaneVQC",
    "PennyLaneKernelSVM",
    "run_pennylane_models",
    "build_feature_map",
    "build_ansatz",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    from quantum.data_prep import prepare

    data = prepare(max_samples=400)
    for result in run_pennylane_models(data, epochs=10, kernel_max_train=150):
        print(result.row())
