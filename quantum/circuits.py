"""
quantum/circuits.py — Feature maps and ansätze, defined once
==============================================================================

WHY BOTH FRAMEWORKS SHARE THIS FILE

    The study compares PennyLane and Qiskit. That comparison only means
    anything if both are running the *same circuit*. If each framework used its
    own convenient built-in — PennyLane's `AngleEmbedding`, Qiskit's `ZZFeatureMap`
    — then a difference in accuracy would tell you nothing about the frameworks,
    because the models would not be the same model.

    So the circuit structure is specified here, framework-neutrally, and each
    backend implements exactly this specification. Matching accuracy across the
    two then becomes a genuine correctness check on the implementations.

THE TWO INGREDIENTS

1.  FEATURE MAP — classical numbers into a quantum state.

    Angle encoding: feature i becomes a rotation angle on qubit i.

        |0> --[ RY(x_i) ]--

    One qubit per feature, so six features need six qubits. Inputs are scaled
    to [0, pi] by `data_prep` so the rotation never wraps and the mapping stays
    injective.

    The optional second-order variant adds entangling rotations on feature
    PRODUCTS:

        --[ RY(x_i) ]--•-----------------•--
                       |                 |
        --[ RY(x_j) ]--X--[ RZ(xi*xj) ]--X--

    This is the ZZ-style map from Havlicek et al. (2019). It creates a feature
    space that is classically hard to simulate — the basis of the claimed
    quantum kernel advantage — but it doubles circuit depth, which on a
    simulator mostly buys runtime.

2.  ANSATZ — the trainable part.

    Layers of parameterised rotations plus a ring of CNOTs:

        --[RY(θ1)][RZ(θ2)]--•---------X--
                            |         |
        --[RY(θ3)][RZ(θ4)]--X--•------|--
                               |      |
        --[RY(θ5)][RZ(θ6)]-----X--•---|--
                                  |   |
        --[RY(θ7)][RZ(θ8)]--------X---•--

    The CNOT ring is what makes the model more than n independent single-qubit
    classifiers: without entanglement the circuit cannot represent correlations
    between features, and Mental Fatigue x Resource Allocation is exactly the
    interaction that predicts burnout here.

DEPTH IS DELIBERATELY SHALLOW
    Two layers by default. Deeper is tempting but runs into BARREN PLATEAUS:
    gradient variance decays exponentially with qubit count and depth
    (McClean et al., 2018), so a deep circuit trains to a flat loss landscape
    and learns nothing. Shallow-and-trainable beats deep-and-stuck.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

FeatureMapKind = Literal["angle", "zz"]
AnsatzKind = Literal["basic", "strong"]


@dataclass(frozen=True)
class CircuitSpec:
    """Framework-neutral description of a variational circuit.

    Both `pennylane_models.py` and `qiskit_models.py` build from this, so the
    two implementations cannot silently diverge.
    """

    n_qubits: int
    n_layers: int = 2
    feature_map: FeatureMapKind = "angle"
    ansatz: AnsatzKind = "basic"
    entanglement: str = "ring"

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters.

        ``basic``  : 2 rotations (RY, RZ) per qubit per layer
        ``strong`` : 3 rotations (RX, RY, RZ) per qubit per layer
        """
        per_qubit = 3 if self.ansatz == "strong" else 2
        return self.n_layers * self.n_qubits * per_qubit

    @property
    def depth_estimate(self) -> int:
        """Rough two-qubit-gate depth, used to warn about barren plateaus."""
        map_depth = 1 if self.feature_map == "angle" else 3
        return map_depth + self.n_layers * 2

    def entangling_pairs(self) -> List[Tuple[int, int]]:
        """Qubit pairs for the entangling layer.

        ``ring``   : 0-1, 1-2, ..., (n-1)-0. Every qubit is entangled with two
                     neighbours at linear gate cost. The wrap-around closes the
                     loop so information can propagate all the way round.
        ``linear`` : as above without the wrap-around.
        ``full``   : every pair. O(n^2) gates; included for completeness, and
                     realistically too deep to train past a few qubits.
        """
        if self.n_qubits < 2:
            return []

        if self.entanglement == "full":
            return [
                (i, j)
                for i in range(self.n_qubits)
                for j in range(i + 1, self.n_qubits)
            ]

        pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        if self.entanglement == "ring" and self.n_qubits > 2:
            pairs.append((self.n_qubits - 1, 0))
        return pairs

    def describe(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "feature_map": self.feature_map,
            "ansatz": self.ansatz,
            "entanglement": self.entanglement,
            "n_parameters": self.n_parameters,
            "estimated_depth": self.depth_estimate,
            "hilbert_space_dimension": 2 ** self.n_qubits,
        }

    def barren_plateau_warning(self) -> str | None:
        """Flag configurations likely to train poorly.

        Gradient variance falls roughly as 1/2^n with qubit count for deep
        random circuits. Past about 10 qubits with several layers, gradients
        are numerically indistinguishable from zero and training silently
        stalls at chance accuracy — which looks like "quantum does not work"
        rather than "this circuit was untrainable".
        """
        if self.n_qubits >= 10 and self.n_layers >= 4:
            return (
                f"{self.n_qubits} qubits x {self.n_layers} layers is deep "
                f"enough to risk barren plateaus: gradients vanish roughly as "
                f"1/2^n, so training may stall at chance accuracy. Reduce "
                f"layers or qubits if the loss does not move."
            )
        return None


#: The configuration used for the headline benchmark. Six qubits matches the
#: six encoded features exactly, so no PCA is needed.
DEFAULT_SPEC = CircuitSpec(n_qubits=6, n_layers=2, feature_map="angle",
                           ansatz="basic", entanglement="ring")


def parameter_shape(spec: CircuitSpec) -> Tuple[int, int, int]:
    """Shape of the trainable parameter tensor: (layers, qubits, rotations)."""
    return (spec.n_layers, spec.n_qubits, 3 if spec.ansatz == "strong" else 2)


__all__ = ["CircuitSpec", "DEFAULT_SPEC", "parameter_shape",
           "FeatureMapKind", "AnsatzKind"]
