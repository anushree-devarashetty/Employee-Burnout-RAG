"""Quantum kernel execution on Qniverse (C-DAC's national quantum platform).

WHAT THIS IS
    `pennylane_models.py` computes the fidelity kernel on a local state-vector
    simulator, analytically — exact overlaps, no sampling. This module computes
    the SAME kernel on Qniverse, where each entry is ESTIMATED from a finite
    number of measurement shots on a remote simulator.

    Comparing the two answers the question our simulator-only results cannot:
    how much does the kernel degrade when fidelity is sampled rather than
    computed? That gap is the headline result of running here.

WHY THIS IS A SEPARATE MODULE
    The Qniverse QSDK is gate-level only. It has QniverseCircuit, gates and a
    job API — there is no VQC or QSVC class, so qiskit-machine-learning does
    not port across. The fidelity kernel does, because it is only ever
    "build a circuit, measure, count the all-zero outcomes".

    The variational classifier deliberately is NOT ported. Training it would
    need one remote job per parameter per gradient step — thousands of jobs
    for a model that already loses to Logistic Regression locally. The kernel
    is the part worth spending credits on.

THE COST MODEL, WHICH DRIVES EVERY DESIGN DECISION HERE
    One circuit = one remote job = credits. A Gram matrix over n training rows
    needs n(n-1)/2 jobs, plus n_test x n_train for prediction. At n=400 that is
    roughly 80,000 jobs, which is not happening. So:

      - The diagonal is never submitted. K(x,x) = 1 exactly, by definition.
      - Only the upper triangle of the symmetric train block is submitted.
      - Every result is cached to disk by content hash. Re-running never
        resubmits a pair that has already been paid for.
      - `--dry-run` builds and counts everything WITHOUT submitting, so the
        job count and credit cost are known before any are spent.

    Treat the cache as the valuable artifact. Delete it and you pay again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - script-mode bootstrap, mirrors the sibling modules
    import sys as _sys
    from pathlib import Path as _Path
except ImportError:  # pragma: no cover
    pass

import numpy as np

from quantum import REPO_ROOT
from quantum.circuits import CircuitSpec, DEFAULT_SPEC

logger = logging.getLogger(__name__)

#: Where kernel entries are cached. One JSON file per (backend, map, shots)
#: configuration, so changing shots does not silently reuse the wrong numbers.
CACHE_DIR = REPO_ROOT / "outputs" / "quantum" / "qniverse_cache"


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class QniverseConfig:
    """Backend selection, in the order the QSDK requires.

    The QSDK enforces processor -> hpc -> simulator and raises ValueError if
    they are set out of order, so they are kept together here and applied in
    sequence by `_build_backend`.

    Defaults target the GPU state-vector simulator on the QACC cluster, which
    is the closest remote analogue to the local `default.qubit` device — that
    makes the local-versus-platform comparison a comparison of SAMPLING, not
    of two unrelated simulators.
    """

    processor: str = "GPU"
    hpc: str = "QACC_Cluster"
    simulator: str = "aer_simulator_statevector_gpu"
    shots: int = 1024
    job_prefix: str = "burnout-kernel"

    #: Seconds between job status polls, and the ceiling on total wait.
    poll_seconds: float = 2.0
    timeout_seconds: float = 600.0

    def slug(self) -> str:
        return f"{self.processor}_{self.hpc}_{self.simulator}_{self.shots}"


def _read_env(key: str) -> str:
    """Read a key from the environment, falling back to the .env file."""
    value = os.environ.get(key, "").strip()
    if value:
        return value

    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        prefix = f"{key}="
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def load_tokens(explicit: Optional[Sequence[str]] = None) -> List[str]:
    """Resolve every available Qniverse API key, in rotation order.

    WHY A LIST AND NOT ONE KEY
        Credits are allocated PER ACCOUNT and cannot be pooled. A team with
        four accounts at 50 credits each has 200 credits in total but a hard
        ceiling of 50 on any single one. Since a Gram matrix over even 16
        rows needs more than 50 jobs, an experiment that cannot move between
        accounts simply cannot be run.

        So the tokens are a queue. When one account is exhausted the runner
        rotates to the next and carries on. The cache is shared across all of
        them, because a kernel entry is a kernel entry regardless of which
        account paid for it.

    ACCEPTED FORMS in .env (first match wins)
        QNIVERSE_TOKENS=key1,key2,key3,key4      <- preferred
        QNIVERSE_TOKEN_1=key1                    <- one per line
        QNIVERSE_TOKEN_2=key2
        QNIVERSE_TOKEN=key1                      <- single-account fallback
    """
    if explicit:
        cleaned = [t.strip() for t in explicit if t and t.strip()]
        if cleaned:
            return cleaned

    joined = _read_env("QNIVERSE_TOKENS")
    if joined:
        tokens = [t.strip() for t in joined.split(",") if t.strip()]
        if tokens:
            return tokens

    numbered: List[str] = []
    for index in range(1, 17):
        value = _read_env(f"QNIVERSE_TOKEN_{index}")
        if value:
            numbered.append(value)
    if numbered:
        return numbered

    single = _read_env("QNIVERSE_TOKEN")
    if single and single != "replace-me":
        return [single]

    raise RuntimeError(
        "No Qniverse token found.\n"
        "  1. Log in at https://qniverse.in and copy the Qniverse API Key "
        "from each account's profile dashboard.\n"
        "  2. Add them to .env as a comma-separated list:\n"
        "       QNIVERSE_TOKENS=key1,key2,key3,key4\n"
        "     Credits are per-account and cannot be pooled, so listing every "
        "account you have raises the ceiling for a single experiment."
    )


def load_token(explicit: Optional[str] = None) -> str:
    """First available token. Kept for callers that only need one."""
    return load_tokens([explicit] if explicit else None)[0]


def _require_qniverse():
    """Import the QSDK, with an actionable message if it is absent.

    Deferred exactly like pennylane and qiskit elsewhere in this package: the
    backend, dashboard and local benchmark must all keep working on a machine
    that has never heard of Qniverse.
    """
    try:
        import Qniverse  # noqa: F401
        from Qniverse.backend import Backend
        from Qniverse.QniverseCircuit import QniverseCircuit, QuantumRegister
        from Qniverse.gates import ry, rz, h, cx, measure  # noqa: F401
        from Qniverse.api import run, fetch_job_result, job_status, credit_balance

        return {
            "Backend": Backend,
            "QniverseCircuit": QniverseCircuit,
            "QuantumRegister": QuantumRegister,
            "gates": {"ry": ry, "rz": rz, "h": h, "cx": cx, "measure": measure},
            "run": run,
            "fetch_job_result": fetch_job_result,
            "job_status": job_status,
            "credit_balance": credit_balance,
        }
    except ImportError as exc:
        raise ImportError(
            "The Qniverse SDK is not installed, so remote execution is "
            "unavailable.\n"
            "  Fix: pip install qniverse\n"
            "  The local simulator benchmark (quantum/benchmark.py) does not "
            "need it and is unaffected."
        ) from exc


# ==============================================================================
# Circuit construction
# ==============================================================================
def build_kernel_circuit(x1: np.ndarray, x2: np.ndarray, spec: CircuitSpec):
    """Circuit whose all-zero measurement probability is |<phi(x1)|phi(x2)>|^2.

    Encode x1, then apply the INVERSE encoding of x2. If the two samples are
    identical the two halves cancel exactly and every shot returns |00...0>.
    The more they differ, the fewer all-zero outcomes — so the all-zero count
    divided by shots IS the kernel value.

    The gate sequences below mirror `pennylane_models.build_feature_map`
    exactly. If that function changes, this one must change with it, or the
    local and remote kernels stop being the same object and the comparison
    becomes meaningless.
    """
    sdk = _require_qniverse()
    QniverseCircuit = sdk["QniverseCircuit"]
    QuantumRegister = sdk["QuantumRegister"]
    g = sdk["gates"]

    n = spec.n_qubits
    qc = QniverseCircuit()
    q = QuantumRegister(n, "q")
    qc.add_register(q)

    if spec.feature_map == "zz":
        pairs = spec.entangling_pairs()

        # --- forward: ZZ map on x1 -------------------------------------------
        for w in range(n):
            qc.add_gate(g["h"], q[w])
            qc.add_gate(g["rz"](float(x1[w])), q[w])
        for a, b in pairs:
            qc.add_gate(g["cx"], q[a], q[b])
            qc.add_gate(g["rz"](float(x1[a] * x1[b])), q[b])
            qc.add_gate(g["cx"], q[a], q[b])

        # --- adjoint: reverse the order and negate every angle ---------------
        # H is its own inverse, so it needs no sign change — only repositioning
        # to the end. CX is also self-inverse.
        for a, b in reversed(pairs):
            qc.add_gate(g["cx"], q[a], q[b])
            qc.add_gate(g["rz"](float(-x2[a] * x2[b])), q[b])
            qc.add_gate(g["cx"], q[a], q[b])
        for w in reversed(range(n)):
            qc.add_gate(g["rz"](float(-x2[w])), q[w])
            qc.add_gate(g["h"], q[w])
    else:
        # Angle map: one RY per qubit, no entanglement. The adjoint of RY(t)
        # is simply RY(-t), so the whole pair circuit is two rotations per
        # qubit and the kernel factorises to prod_i cos^2((x1_i - x2_i)/2).
        #
        # That factorisation is worth stating in the report: this map is
        # classically computable in O(n), which is precisely why it cannot
        # beat a classical RBF. Use --feature-map zz for the entangling map.
        for w in range(n):
            qc.add_gate(g["ry"](float(x1[w])), q[w])
        for w in range(n):
            qc.add_gate(g["ry"](float(-x2[w])), q[w])

    qc.measure_all()
    return qc


# ==============================================================================
# Result parsing
# ==============================================================================
def _extract_counts(payload) -> Dict[str, int]:
    """Pull a {bitstring: count} mapping out of whatever fetch_job_result returns.

    The QSDK docs specify the call but not the response shape, and it differs
    between simulators. Rather than guess one layout and fail cryptically on
    the others, try the plausible ones and, failing that, raise an error that
    prints the actual payload so the mapping can be corrected in one place.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            pass

    if isinstance(payload, dict):
        # Direct mapping: {"000000": 812, "000001": 44, ...}
        if payload and all(isinstance(v, (int, float)) for v in payload.values()):
            keys = list(payload.keys())
            if all(set(str(k)) <= {"0", "1", " "} for k in keys):
                return {str(k).replace(" ", ""): int(v) for k, v in payload.items()}

        # Nested under a conventional key.
        for key in ("counts", "result", "results", "data", "histogram",
                    "measurement_counts", "output"):
            if key in payload:
                nested = _extract_counts(payload[key])
                if nested:
                    return nested

    if isinstance(payload, list) and payload:
        return _extract_counts(payload[0])

    raise ValueError(
        "Could not find a counts dictionary in the Qniverse job result.\n"
        f"  Received type {type(payload).__name__}: {str(payload)[:400]}\n"
        "  Fix: adjust _extract_counts() in quantum/qniverse_backend.py to "
        "match the shape your simulator returns."
    )


def kernel_value_from_counts(counts: Dict[str, int], n_qubits: int,
                             shots: int) -> float:
    """P(all zeros) — the fidelity estimate."""
    zero = "0" * n_qubits
    total = sum(counts.values()) or shots
    return float(counts.get(zero, 0)) / float(total)


# ==============================================================================
# The runner
# ==============================================================================
@dataclass
class QniverseKernel:
    """Computes fidelity kernel matrices by submitting jobs to Qniverse."""

    spec: CircuitSpec = field(default_factory=lambda: DEFAULT_SPEC)
    config: QniverseConfig = field(default_factory=QniverseConfig)
    token: Optional[str] = None
    tokens: Optional[List[str]] = None
    dry_run: bool = False

    #: Jobs to draw from one account before rotating to the next. Set this to
    #: the per-account credit allowance. Rotation also happens automatically
    #: when the platform reports the account is out of credit, so this is a
    #: belt-and-braces measure rather than the only defence.
    per_token_budget: int = 50

    #: Hard ceiling on submissions for the whole run. Nothing is submitted
    #: beyond this, whatever the token budgets say. The cheapest insurance
    #: against a loop bug emptying every account.
    max_jobs: Optional[int] = None

    _cache: Dict[str, float] = field(default_factory=dict, init=False)
    _cache_path: Optional[Path] = field(default=None, init=False)
    _backend = None
    _token_list: List[str] = field(default_factory=list, init=False)
    _token_index: int = field(default=0, init=False)
    _token_jobs: Dict[int, int] = field(default_factory=dict, init=False)
    _exhausted: set = field(default_factory=set, init=False)
    _submitted: int = field(default=0, init=False)
    _cache_hits: int = field(default=0, init=False)
    _planned: int = field(default=0, init=False)

    # ----------------------------------------------------------------- tokens
    #: Accounts assumed when planning without credentials configured. A dry
    #: run must work before the keys are in .env — costing an experiment is
    #: exactly the thing you want to do BEFORE setting up access.
    assume_accounts: int = 4

    def _ensure_tokens(self) -> List[str]:
        if not self._token_list:
            explicit = self.tokens or ([self.token] if self.token else None)
            try:
                self._token_list = load_tokens(explicit)
                logger.info("Loaded %d Qniverse account token(s); budget %d "
                            "job(s) each = %d job(s) total",
                            len(self._token_list), self.per_token_budget,
                            len(self._token_list) * self.per_token_budget)
            except RuntimeError:
                if not self.dry_run:
                    raise
                # Planning only. These are never used to submit anything.
                self._token_list = ["<unconfigured>"] * self.assume_accounts
                logger.info("[dry-run] no tokens configured; assuming %d "
                            "account(s) for planning", self.assume_accounts)
        return self._token_list

    @property
    def budget(self) -> int:
        """Total jobs available across every configured account."""
        return len(self._ensure_tokens()) * self.per_token_budget

    def _current_token(self) -> str:
        tokens = self._ensure_tokens()
        while self._token_index < len(tokens):
            if self._token_index in self._exhausted:
                self._token_index += 1
                continue
            if self._token_jobs.get(self._token_index, 0) >= self.per_token_budget:
                logger.info("Account %d reached its %d-job budget; rotating.",
                            self._token_index + 1, self.per_token_budget)
                self._exhausted.add(self._token_index)
                self._backend = None          # backend is bound to the token
                self._token_index += 1
                continue
            return tokens[self._token_index]

        raise RuntimeError(
            f"All {len(tokens)} Qniverse account(s) are exhausted after "
            f"{self._submitted} job(s).\n"
            f"  Everything completed so far is cached, so re-running with more "
            f"accounts in QNIVERSE_TOKENS resumes rather than restarts.\n"
            f"  Cache: {self._cache_path}"
        )

    def _rotate(self, reason: str) -> None:
        """Mark the current account unusable and move to the next."""
        logger.warning("Account %d unusable (%s); rotating.",
                       self._token_index + 1, reason)
        self._exhausted.add(self._token_index)
        self._backend = None
        self._token_index += 1

    # ----------------------------------------------------------------- cache
    def _load_cache(self) -> None:
        if self._cache_path is not None:
            return
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{self.config.slug()}_{self.spec.feature_map}_{self.spec.n_qubits}q.json"
        self._cache_path = CACHE_DIR / name
        if self._cache_path.exists():
            self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            logger.info("Loaded %d cached kernel entries from %s",
                        len(self._cache), self._cache_path.name)

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    @staticmethod
    def _pair_key(x1: np.ndarray, x2: np.ndarray) -> str:
        """Content hash of the unordered pair.

        Rounded to 6 decimals so floating-point noise in the scaler cannot
        produce a cache miss for what is numerically the same pair. Sorted so
        K(a,b) and K(b,a) share one entry — the kernel is symmetric, and
        paying twice for it would be careless.
        """
        a = np.round(np.asarray(x1, dtype=float), 6).tobytes()
        b = np.round(np.asarray(x2, dtype=float), 6).tobytes()
        lo, hi = (a, b) if a <= b else (b, a)
        return hashlib.sha256(lo + hi).hexdigest()[:32]

    # --------------------------------------------------------------- backend
    def _build_backend(self):
        """Backend for the CURRENT account.

        Rebuilt whenever the token rotates, because the QSDK binds the key at
        Backend() construction time — reusing a backend built against an
        exhausted account would keep failing.
        """
        if self._backend is not None:
            return self._backend

        sdk = _require_qniverse()
        tok = self._current_token()

        # The QSDK reads `token` from the caller's GLOBAL scope during
        # Backend() construction and raises RuntimeError if it is absent.
        # Injecting it here keeps the key out of the module source.
        globals()["token"] = tok

        backend = sdk["Backend"]()
        backend.processor(self.config.processor)
        backend.hpc(self.config.hpc)
        backend.simulator(self.config.simulator)
        self._backend = backend
        logger.info("Backend ready on account %d: %s / %s / %s, %d shots",
                    self._token_index + 1, self.config.processor,
                    self.config.hpc, self.config.simulator, self.config.shots)
        return backend

    def credit_balance(self, index: Optional[int] = None):
        """Credit balance for one account, or all of them."""
        sdk = _require_qniverse()
        tokens = self._ensure_tokens()
        if index is not None:
            return sdk["credit_balance"](tokens[index])
        balances = {}
        for i, tok in enumerate(tokens):
            try:
                balances[f"account_{i + 1}"] = sdk["credit_balance"](tok)
            except Exception as exc:  # one bad key must not hide the others
                balances[f"account_{i + 1}"] = f"unavailable ({exc})"
        return balances

    # ------------------------------------------------------------ single job
    #: Substrings that mean "this account cannot pay", not "the circuit is
    #: wrong". Matched case-insensitively against the exception text. Rotating
    #: on these is the difference between using 200 credits and using 50.
    _CREDIT_ERRORS = ("credit", "quota", "insufficient", "limit exceeded",
                      "balance", "exhaust", "unauthor", "forbidden", "401",
                      "403", "payment")

    def _run_pair(self, x1: np.ndarray, x2: np.ndarray, label: str) -> float:
        sdk = _require_qniverse()
        qc = build_kernel_circuit(x1, x2, self.spec)

        # One attempt per remaining account. A credit failure rotates and
        # retries the SAME pair, so no work is lost at an account boundary.
        attempts = len(self._ensure_tokens())
        last_error: Optional[Exception] = None

        for _ in range(attempts):
            backend = self._build_backend()
            token_index = self._token_index
            try:
                job = sdk["run"](
                    qc, backend, self.config.shots,
                    token=self._current_token(),
                    job_name=f"{self.config.job_prefix}-{label}",
                )
            except Exception as exc:
                message = str(exc).lower()
                if any(flag in message for flag in self._CREDIT_ERRORS):
                    self._rotate(f"submit rejected: {exc}")
                    last_error = exc
                    continue
                raise

            self._submitted += 1
            self._token_jobs[token_index] = self._token_jobs.get(token_index, 0) + 1
            break
        else:
            raise RuntimeError(
                f"Every account refused job {label}. Last error: {last_error}"
            )

        deadline = time.time() + self.config.timeout_seconds
        while time.time() < deadline:
            try:
                status = str(job.job_status()).lower()
            except Exception:  # status endpoint hiccup is not fatal; retry
                status = ""
            if any(s in status for s in ("complete", "success", "done", "finish")):
                break
            if any(s in status for s in ("fail", "error", "cancel")):
                raise RuntimeError(f"Qniverse job {label} ended as: {status}")
            time.sleep(self.config.poll_seconds)
        else:
            raise TimeoutError(
                f"Job {label} did not finish within "
                f"{self.config.timeout_seconds:.0f}s. It may still be queued — "
                f"re-run to resume; completed pairs are cached."
            )

        counts = _extract_counts(job.fetch_job_result())
        return kernel_value_from_counts(counts, self.spec.n_qubits,
                                        self.config.shots)

    # ------------------------------------------------------------ gram matrix
    def gram(self, a: np.ndarray, b: Optional[np.ndarray] = None,
             tag: str = "K") -> np.ndarray:
        """Kernel matrix between `a` and `b` (or `a` with itself if b is None).

        Symmetric case skips the diagonal (exactly 1) and the lower triangle.
        """
        self._load_cache()
        symmetric = b is None
        b = a if symmetric else b

        matrix = np.zeros((len(a), len(b)), dtype=float)
        jobs: List[Tuple[int, int, str]] = []

        for i in range(len(a)):
            for j in range(len(b)):
                if symmetric and j < i:
                    continue
                if symmetric and i == j:
                    matrix[i, j] = 1.0
                    continue

                key = self._pair_key(a[i], b[j])
                if key in self._cache:
                    value = self._cache[key]
                    matrix[i, j] = value
                    if symmetric:
                        matrix[j, i] = value
                    self._cache_hits += 1
                else:
                    jobs.append((i, j, key))

        self._planned += len(jobs)

        if self.dry_run:
            logger.info("[dry-run] %s would submit %d job(s); %d already cached",
                        tag, len(jobs), self._cache_hits)
            return matrix

        for count, (i, j, key) in enumerate(jobs, start=1):
            if self.max_jobs is not None and self._submitted >= self.max_jobs:
                logger.warning(
                    "Stopping at the --max-jobs ceiling of %d. %d pair(s) in "
                    "'%s' were not computed; everything done so far is cached, "
                    "so re-running resumes from here.",
                    self.max_jobs, len(jobs) - count + 1, tag)
                self._save_cache()
                break

            value = self._run_pair(a[i], b[j], f"{tag}-{i}-{j}")
            self._cache[key] = value
            matrix[i, j] = value
            if symmetric:
                matrix[j, i] = value

            if count % 10 == 0 or count == len(jobs):
                self._save_cache()
                logger.info("  %s: %d/%d jobs complete", tag, count, len(jobs))

        self._save_cache()
        return matrix

    # ------------------------------------------------------------- reporting
    def stats(self) -> Dict[str, object]:
        return {
            "jobs_submitted": self._submitted,
            "jobs_planned": self._planned,
            "cache_hits": self._cache_hits,
            "cached_total": len(self._cache),
            "accounts_configured": len(self._token_list) or None,
            "accounts_exhausted": len(self._exhausted),
            "jobs_per_account": {f"account_{i + 1}": n
                                 for i, n in sorted(self._token_jobs.items())},
        }


def plan_budget(total_credits: int) -> List[Tuple[int, int, int]]:
    """Feasible (train, test, jobs) splits that fit inside a credit budget.

    Jobs needed = n(n-1)/2 for the symmetric training block, plus m*n for the
    test block. The diagonal is free because K(x,x) = 1 by definition.

    Returned largest-first, so the head of the list is the biggest experiment
    the budget allows.
    """
    options: List[Tuple[int, int, int]] = []
    for n in range(6, 61):
        train_jobs = n * (n - 1) // 2
        if train_jobs >= total_credits:
            break
        for m in range(3, n + 1):
            jobs = train_jobs + m * n
            if jobs <= total_credits:
                options.append((n, m, jobs))
    options.sort(key=lambda t: (-(t[0] + t[1]), -t[2]))
    return options


# ==============================================================================
# Numerical repair
# ==============================================================================
def make_psd(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Force a sampled Gram matrix back to positive semi-definite.

    An exactly computed fidelity kernel is PSD by construction: it equals
    Tr[rho(x) rho(y)], an inner product between density matrices. Estimating it
    from a finite number of shots adds independent noise to each entry, which
    can push the smallest eigenvalues slightly negative — and sklearn's SVC
    will either warn or produce a meaningless model when that happens.

    Clipping negative eigenvalues to zero is the standard repair. The size of
    the clip is itself informative: a large correction means the shot count is
    too low for the kernel to be resolved, and is worth reporting.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    negative = float(values[values < -tol].sum())
    if negative < -tol:
        logger.warning(
            "Sampled kernel had negative eigenvalues summing to %.2e — "
            "clipping to zero. Increase shots if this is large.", negative)
    repaired = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    return repaired


__all__ = [
    "QniverseConfig",
    "QniverseKernel",
    "build_kernel_circuit",
    "kernel_value_from_counts",
    "load_token",
    "make_psd",
    "CACHE_DIR",
]
