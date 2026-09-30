"""Run the quantum kernel on Qniverse and compare it against local simulation.

THE EXPERIMENT
    Take a small subset of rows. Compute the fidelity kernel twice:

      1. LOCALLY, analytically, on PennyLane's state-vector simulator. Exact
         overlaps, no sampling.
      2. REMOTELY, on Qniverse, where each entry is estimated from a finite
         number of measurement shots.

    Fit the same SVC on both Gram matrices and report both accuracies, plus
    the elementwise difference between the two kernels.

WHY THIS IS THE RESULT WORTH HAVING
    Every quantum number in the project so far comes from an exact simulator,
    which is the single most obvious thing for a reviewer to challenge: real
    devices sample, they do not compute overlaps in closed form. This produces
    the missing number — how much accuracy is lost when fidelity is estimated
    rather than calculated — and it does so on India's national quantum
    platform rather than a laptop.

    Expect the platform accuracy to be EQUAL OR WORSE, never better. Sampling
    adds noise; it does not add information. A small gap is the good outcome:
    it says the kernel survives realistic measurement.

SCOPE, AND WHY IT IS SMALL
    One circuit is one remote job, and jobs cost credits. The training block
    alone needs n(n-1)/2 jobs. The defaults below are deliberately tiny — this
    is a proof of execution and a noise measurement, not a re-run of the full
    benchmark. Check the job count with --dry-run before spending anything.

USAGE
    # 1. See what it would cost. Submits nothing.
    python quantum/run_qniverse.py --dry-run

    # 2. Actually run it.
    python quantum/run_qniverse.py --train-rows 16 --test-rows 8 --shots 1024

    # 3. The entangling map — the one that is not classically trivial.
    python quantum/run_qniverse.py --feature-map zz --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - allows `python quantum/run_qniverse.py`
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
except Exception:  # pragma: no cover
    pass

import numpy as np

from quantum import REPO_ROOT
from quantum.circuits import CircuitSpec
from quantum.data_prep import prepare
from quantum.qniverse_backend import (
    QniverseConfig,
    QniverseKernel,
    make_psd,
    plan_budget,
)

logger = logging.getLogger("qniverse")

OUT_DIR = REPO_ROOT / "outputs" / "quantum"


# ==============================================================================
# Local reference kernel
# ==============================================================================
def local_kernel(a: np.ndarray, b: Optional[np.ndarray], spec: CircuitSpec):
    """The same kernel, computed exactly on the local simulator.

    Reuses PennyLaneKernelSVM's own Gram routine rather than reimplementing it,
    so the reference cannot drift from what the main benchmark reports.
    """
    from quantum.pennylane_models import PennyLaneKernelSVM

    model = PennyLaneKernelSVM(spec=spec)
    return model._gram(a, a if b is None else b)


# ==============================================================================
# Experiment
# ==============================================================================
def run_experiment(args: argparse.Namespace) -> Dict:
    spec = CircuitSpec(
        n_qubits=args.qubits,
        n_layers=args.layers,
        feature_map=args.feature_map,
        ansatz="basic",
        entanglement="ring",
    )
    config = QniverseConfig(
        processor=args.processor,
        hpc=args.hpc,
        simulator=args.simulator,
        shots=args.shots,
    )

    # --- identical rows to the local benchmark -------------------------------
    logger.info("Preparing data (same pipeline as benchmark.py)...")
    data = prepare(n_qubits=args.qubits, max_samples=args.samples,
                   random_state=args.seed)

    x_train = np.asarray(data.x_train)[: args.train_rows]
    y_train = np.asarray(data.y_train)[: args.train_rows]
    x_test = np.asarray(data.x_test)[: args.test_rows]
    y_test = np.asarray(data.y_test)[: args.test_rows]

    n_tr, n_te = len(x_train), len(x_test)
    train_jobs = n_tr * (n_tr - 1) // 2
    test_jobs = n_te * n_tr
    total_jobs = train_jobs + test_jobs

    print()
    print("=" * 68)
    print("  QNIVERSE QUANTUM KERNEL RUN")
    print("=" * 68)
    print(f"  Backend      : {config.processor} / {config.hpc}")
    print(f"  Simulator    : {config.simulator}")
    print(f"  Feature map  : {spec.feature_map}   ({spec.n_qubits} qubits)")
    print(f"  Shots        : {config.shots}")
    print(f"  Train / test : {n_tr} / {n_te} rows")
    print("-" * 68)
    print(f"  Train Gram   : {train_jobs} jobs  (upper triangle, diagonal free)")
    print(f"  Test Gram    : {test_jobs} jobs")
    print(f"  TOTAL        : {total_jobs} remote jobs")
    print("=" * 68)
    print()

    if args.dry_run:
        print("DRY RUN — nothing was submitted and no credits were spent.")
        print("Re-run without --dry-run to execute.\n")

    kernel = QniverseKernel(
        spec=spec, config=config, dry_run=args.dry_run,
        per_token_budget=args.per_account, max_jobs=args.max_jobs,
    )

    budget = kernel.budget
    print(f"  Accounts     : {len(kernel._ensure_tokens())} "
          f"x {args.per_account} credits = {budget} jobs available")
    if total_jobs > budget:
        print()
        print(f"  !! This run needs {total_jobs} jobs but only {budget} "
              f"are available.")
        print("     Feasible splits within budget:")
        for n, m, j in plan_budget(budget)[:5]:
            print(f"        --train-rows {n:<3} --test-rows {m:<3}  "
                  f"({j} jobs)")
        print()
        if not args.dry_run:
            raise RuntimeError(
                "Refusing to start a run that cannot finish. Pick a split "
                "from the list above, or raise --per-account / add more "
                "tokens to QNIVERSE_TOKENS.")


    if not args.dry_run:
        try:
            for name, bal in kernel.credit_balance().items():
                print(f"  {name}: {bal}")
        except Exception as exc:  # non-fatal; do not block the run
            logger.warning("Could not read credit balance: %s", exc)

    # --- remote Gram matrices ------------------------------------------------
    started = time.time()
    k_train = kernel.gram(x_train, None, tag="train")
    k_test = kernel.gram(x_test, x_train, tag="test")
    remote_seconds = time.time() - started

    if args.dry_run:
        return {"dry_run": True, "planned_jobs": total_jobs,
                **kernel.stats()}

    # --- local reference on the SAME rows ------------------------------------
    logger.info("Computing the local analytic kernel for comparison...")
    started = time.time()
    k_train_local = local_kernel(x_train, None, spec)
    k_test_local = local_kernel(x_test, x_train, spec)
    local_seconds = time.time() - started

    # --- fit the identical classifier on both --------------------------------
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.svm import SVC

    def fit_and_score(ktr, kte, repair: bool):
        gram = make_psd(ktr) if repair else ktr
        model = SVC(kernel="precomputed", C=1.0, random_state=42)
        model.fit(gram, y_train)
        pred = model.predict(kte)
        return (float(accuracy_score(y_test, pred)),
                float(f1_score(y_test, pred, average="weighted")))

    remote_acc, remote_f1 = fit_and_score(k_train, k_test, repair=True)
    local_acc, local_f1 = fit_and_score(k_train_local, k_test_local, repair=False)

    # --- how far apart are the two kernels? ----------------------------------
    diff_train = np.abs(k_train - k_train_local)
    diff_test = np.abs(k_test - k_test_local)
    all_diff = np.concatenate([diff_train.ravel(), diff_test.ravel()])

    # Sampling error on a proportion from N shots is ~sqrt(p(1-p)/N); at
    # p=0.5 and 1024 shots that is about 0.0156. Comparing the observed mean
    # difference against that tells you whether the gap is just shot noise
    # or something structural.
    expected_noise = float(np.sqrt(0.25 / config.shots))

    results = {
        "dry_run": False,
        "backend": {
            "processor": config.processor,
            "hpc": config.hpc,
            "simulator": config.simulator,
            "shots": config.shots,
        },
        "circuit": spec.describe(),
        "rows": {"train": n_tr, "test": n_te},
        "jobs": kernel.stats(),
        "accuracy": {
            "qniverse_sampled": remote_acc,
            "local_analytic": local_acc,
            "gap": local_acc - remote_acc,
        },
        "f1": {"qniverse_sampled": remote_f1, "local_analytic": local_f1},
        "kernel_difference": {
            "mean_abs": float(all_diff.mean()),
            "max_abs": float(all_diff.max()),
            "expected_shot_noise": expected_noise,
            "ratio_to_expected": float(all_diff.mean() / expected_noise),
        },
        "timing_seconds": {
            "qniverse_wall_clock": remote_seconds,
            "local_analytic": local_seconds,
        },
    }

    # --- report --------------------------------------------------------------
    print()
    print("=" * 68)
    print("  RESULTS")
    print("=" * 68)
    print(f"  Accuracy, Qniverse (sampled)   : {remote_acc:.4f}")
    print(f"  Accuracy, local (analytic)     : {local_acc:.4f}")
    print(f"  Gap                            : {local_acc - remote_acc:+.4f}")
    print("-" * 68)
    print(f"  Mean |K_qniverse - K_local|    : {all_diff.mean():.4f}")
    print(f"  Max  |K_qniverse - K_local|    : {all_diff.max():.4f}")
    print(f"  Expected shot noise (1/sqrt4N) : {expected_noise:.4f}")
    print(f"  Observed / expected            : "
          f"{all_diff.mean() / expected_noise:.2f}x")
    print("-" * 68)
    print(f"  Jobs submitted                 : {kernel.stats()['jobs_submitted']}")
    print(f"  Served from cache              : {kernel.stats()['cache_hits']}")
    print(f"  Qniverse wall clock            : {remote_seconds:.1f}s")
    print(f"  Local analytic                 : {local_seconds:.1f}s")
    print("=" * 68)
    print()
    if all_diff.mean() <= 2 * expected_noise:
        print("  Reading: the difference is consistent with pure shot noise.")
        print("  The kernel survives finite sampling at this shot count.")
    else:
        print("  Reading: the difference EXCEEDS shot noise — the remote")
        print("  simulator differs structurally, or shots are too low.")
    print()

    try:
        print("Credit balance after:")
        for name, bal in kernel.credit_balance().items():
            print(f"  {name}: {bal}")
    except Exception:
        pass

    return results


# ==============================================================================
# Entry point
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the quantum kernel on Qniverse and compare against "
                    "local analytic simulation.")
    parser.add_argument("--train-rows", type=int, default=14,
                        help="Training rows. Cost grows as n^2/2. Default 14 "
                             "(fits 4 accounts x 50 credits).")
    parser.add_argument("--test-rows", type=int, default=7,
                        help="Test rows. Cost is n_test x n_train. Default 7.")
    parser.add_argument("--per-account", type=int, default=50,
                        help="Credits per Qniverse account. Default 50.")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Hard ceiling on total submissions. Safety net.")
    parser.add_argument("--plan", action="store_true",
                        help="Print feasible train/test splits for the "
                             "available budget, then exit.")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--feature-map", choices=["angle", "zz"], default="angle")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1500,
                        help="Rows drawn by data_prep before subsetting.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processor", default="GPU")
    parser.add_argument("--hpc", default="QACC_Cluster")
    parser.add_argument("--simulator", default="aer_simulator_statevector_gpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count jobs and estimate cost WITHOUT submitting.")
    parser.add_argument("--out", default=str(OUT_DIR / "qniverse_results.json"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.plan:
        from quantum.qniverse_backend import load_tokens
        try:
            n_acct = len(load_tokens())
        except RuntimeError:
            n_acct = 4
            print("(no tokens configured yet - assuming 4 accounts)")
        budget = n_acct * args.per_account
        print(f"\nBudget: {n_acct} account(s) x {args.per_account} = "
              f"{budget} jobs\n")
        print(f"  {'train':>6} {'test':>6} {'jobs':>7}  {'headroom':>9}")
        for n, m, j in plan_budget(budget)[:12]:
            print(f"  {n:>6} {m:>6} {j:>7}  {budget - j:>9}")
        print()
        return 0

    try:
        results = run_experiment(args)
    except ImportError as exc:
        print(f"\n{exc}\n")
        return 1
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        return 1

    # Save FIRST, print after — a crash in reporting must never lose results
    # that cost credits to obtain.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
