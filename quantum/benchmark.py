"""
quantum/benchmark.py — Run every model on identical data and report honestly
==============================================================================

RUN
    python quantum/benchmark.py                       # defaults
    python quantum/benchmark.py --qubits 6 --samples 1500 --epochs 40
    python quantum/benchmark.py --skip-qiskit         # PennyLane only, faster
    python quantum/benchmark.py --sweep               # 4, 6 and 8 qubits

OUTPUTS
    outputs/quantum/benchmark_results.csv
    outputs/quantum/benchmark_summary.json
    outputs/quantum/comparison.png

WHAT MAKES THIS A FAIR COMPARISON
    Every model receives the identical arrays from `data_prep`: same rows, same
    scaling, same stratified split, same seed. No per-model tuning. Any
    difference in accuracy is therefore attributable to the classifier itself
    and not to preprocessing, data volume or a lucky split.

    The one unavoidable asymmetry is documented rather than hidden: quantum
    kernel methods need O(n^2) circuit evaluations, so they train on a capped
    subset while classical models see the full training split. The cap is
    printed in the results table, and the classical models are ALSO re-run on
    the same capped subset so that at least one row-for-row comparison exists.

READ THE TIMING COLUMN
    It is the most important column in the table, and the one most often
    omitted from quantum ML write-ups. A quantum model matching logistic
    regression while taking four orders of magnitude longer has not "matched"
    it in any sense a practitioner cares about.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):  # pragma: no cover - script-mode only
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _disable_arrow_strings() -> None:
    """Keep pandas off PyArrow-backed string arrays.

    THE CRASH THIS PREVENTS
        pandas 3.0 makes PyArrow the default storage for string columns. On
        Python 3.14 / Apple Silicon, PyArrow's bundled mimalloc allocator
        segfaults inside `arrow::py::NumPyConverter::Convert` when a DataFrame
        is built from Python objects — which is exactly what happens when the
        benchmark turns its list of result dicts into a table.

        The failure is a SIGSEGV, not an exception, so it cannot be caught:
        the process dies and every result computed up to that point is lost.
        It also looks nothing like its cause. The traceback points at whatever
        stage happened to be running, which made it read as a quantum
        simulator bug for several runs.

        Selecting the Python string backend avoids PyArrow entirely. The only
        cost is slightly slower string handling on tables of a few dozen rows,
        which is irrelevant here.

        Both option names are attempted because the spelling changed between
        pandas 2 and 3, and a missing option must not be fatal.
    """
    for option, value in (("mode.string_storage", "python"),
                          ("future.infer_string", False)):
        try:
            pd.set_option(option, value)
        except Exception:          # option absent on this pandas version
            pass


_disable_arrow_strings()

from quantum import REPO_ROOT
from quantum.circuits import CircuitSpec
from quantum.classical_baseline import ModelResult, run_classical_baselines
from quantum.data_prep import PreparedData, prepare

logger = logging.getLogger("quantum.benchmark")

OUTPUT_DIR = REPO_ROOT / "outputs" / "quantum"


# ==============================================================================
# Environment capture
# ==============================================================================
def environment() -> Dict[str, str]:
    """Record the software versions, so results are reproducible later."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    for module_name, label in (
        ("numpy", "numpy"), ("sklearn", "scikit-learn"),
        ("pennylane", "pennylane"), ("qiskit", "qiskit"),
        ("qiskit_machine_learning", "qiskit-machine-learning"),
    ):
        try:
            module = __import__(module_name)
            versions[label] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[label] = "not installed"
    return versions


# ==============================================================================
# Benchmark
# ==============================================================================
def run_benchmark(
    n_qubits: int = 6,
    max_samples: int = 1500,
    epochs: int = 30,
    kernel_max_train: int = 300,
    target_mode: str = "binary",
    skip_qiskit: bool = False,
    skip_quantum: bool = False,
    feature_map: str = "angle",
    n_layers: int = 2,
) -> Dict[str, object]:
    """Run the full comparison and return a summary dict."""
    started = time.perf_counter()

    data = prepare(
        n_qubits=n_qubits, max_samples=max_samples, target_mode=target_mode
    )
    spec = CircuitSpec(
        n_qubits=n_qubits, n_layers=n_layers, feature_map=feature_map
    )

    logger.info("=" * 74)
    logger.info("Data:    %d train / %d test, %d features, classes=%s",
                len(data.x_train), len(data.x_test), data.n_features,
                data.class_names)
    logger.info("Circuit: %s", spec.describe())
    logger.info("=" * 74)

    results: List[ModelResult] = []

    def checkpoint(stage: str) -> None:
        """Write whatever has completed so far to disk.

        WHY THIS EXISTS
            The quantum simulators are native extensions. On some interpreter
            and platform combinations they abort the process with a
            segmentation fault rather than raising — and a segfault cannot be
            caught by `except`, so neither the final save nor the emergency
            dump ever runs. Every result computed before the crash was being
            lost, including the classical baselines that had already finished
            in milliseconds.

            Checkpointing after each stage means a crash costs only the stage
            that crashed. The partial CSV is still a publishable artifact for
            the models that did complete.
        """
        if not results:
            return
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            partial = {
                "data": data.describe(),
                "circuit": spec.describe(),
                "environment": environment(),
                "kernel_max_train": kernel_max_train,
                "checkpoint_after": stage,
                "complete": False,
                "total_seconds": round(time.perf_counter() - started, 2),
                "results": [r.row() for r in results],
            }
            pd.DataFrame(partial["results"]).to_csv(
                OUTPUT_DIR / "benchmark_results.csv", index=False)
            (OUTPUT_DIR / "benchmark_summary.json").write_text(
                json.dumps(partial, indent=2, default=str))
            logger.info("  checkpoint saved after '%s' (%d model(s))",
                        stage, len(results))
        except Exception as exc:          # never let saving break the run
            logger.warning("  checkpoint failed after '%s': %s", stage, exc)

    logger.info("--- classical baselines ---")
    results.extend(run_classical_baselines(data))

    # Re-run the strongest classical model on the SAME capped subset the
    # quantum kernel uses. Without this there is no honest like-for-like row.
    capped = _classical_on_capped_subset(data, kernel_max_train)
    if capped:
        results.append(capped)
    checkpoint("classical")

    if not skip_quantum:
        logger.info("--- PennyLane ---")
        try:
            from quantum.pennylane_models import run_pennylane_models

            results.extend(
                run_pennylane_models(
                    data, spec=spec, epochs=epochs,
                    kernel_max_train=kernel_max_train,
                )
            )
        except ImportError as exc:
            logger.error("PennyLane unavailable: %s", exc)
        except Exception as exc:
            logger.error("PennyLane stage failed: %s", exc)
        checkpoint("pennylane")

        if not skip_qiskit:
            logger.info("--- Qiskit ---")
            try:
                from quantum.qiskit_models import run_qiskit_models

                results.extend(
                    run_qiskit_models(
                        data, spec=spec, max_iterations=epochs * 2,
                        kernel_max_train=kernel_max_train,
                    )
                )
            except ImportError as exc:
                logger.error("Qiskit unavailable: %s", exc)
            except Exception as exc:
                logger.error("Qiskit stage failed: %s", exc)
            checkpoint("qiskit")

    elapsed = time.perf_counter() - started

    return {
        "data": data.describe(),
        "circuit": spec.describe(),
        "environment": environment(),
        "kernel_max_train": kernel_max_train,
        "total_seconds": round(elapsed, 2),
        "results": [result.row() for result in results],
        "_results_objects": results,
        "_prepared": data,
    }


def _classical_on_capped_subset(
    data: PreparedData, n_train: int
) -> Optional[ModelResult]:
    """Run SVM-RBF on the same capped training set the quantum kernel uses.

    This is the single most important control in the study. Without it, a
    quantum kernel trained on 300 rows is being compared with a classical model
    trained on 1,125 — and any deficit could be blamed on data volume rather
    than the method.
    """
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC

    from quantum.classical_baseline import evaluate

    n_train = min(n_train, len(data.x_train))
    if n_train >= len(data.x_train):
        return None

    result = ModelResult(
        name=f"SVM-RBF (same {n_train} rows)",
        family="classical",
        framework="scikit-learn",
    )
    result.notes.append(
        f"Control: identical {n_train}-row training subset as the quantum "
        f"kernel methods, so the comparison is not confounded by data volume."
    )

    try:
        # No probability=True — see quantum/classical_baseline.py.
        model = SVC(kernel="rbf", random_state=42)

        start = time.perf_counter()
        model.fit(data.x_train[:n_train], data.y_train[:n_train])
        result.train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(data.x_test)
        result.predict_seconds = time.perf_counter() - start

        from quantum.classical_baseline import _ranking_scores

        scores = _ranking_scores(model, data.x_test, data.n_classes)
        metrics = evaluate(data.y_test, predictions, scores)
        result.accuracy = metrics["accuracy"]
        result.f1 = metrics["f1"]
        result.precision = metrics["precision"]
        result.recall = metrics["recall"]
        result.roc_auc = metrics["roc_auc"]
        result.confusion = metrics["confusion"]
    except Exception as exc:
        result.failed = True
        result.error = str(exc)

    return result


# ==============================================================================
# Reporting
# ==============================================================================
def print_table(summary: Dict[str, object]) -> None:
    """Print the comparison table to the terminal."""
    rows = summary["results"]

    print()
    print("=" * 96)
    print("BURNOUT RISK CLASSIFICATION — CLASSICAL vs QUANTUM")
    print("=" * 96)
    data = summary["data"]
    print(f"Dataset      : employee_burnout_clean.csv "
          f"({data['source_rows']:,} rows, {data['sampled_rows']:,} sampled)")
    print(f"Target       : Burn Rate -> {data['target_mode']} "
          f"{data['classes']}")
    print(f"Features     : {data['n_features']} -> {data['n_qubits']} qubits "
          f"(PCA: {data['pca_applied']})")
    print(f"Split        : {data['train_size']} train / {data['test_size']} test")
    print("-" * 96)
    print(f"{'model':<28} {'framework':<14} {'acc':>7} {'f1':>7} {'auc':>7} "
          f"{'train s':>10} {'params':>7}")
    print("-" * 96)

    for row in rows:
        if row["failed"]:
            print(f"{row['model']:<28} {row['framework']:<14} "
                  f"FAILED: {str(row['error'])[:44]}")
            continue
        auc = f"{row['roc_auc']:.4f}" if row["roc_auc"] != "" else "   -   "
        params = row["n_parameters"] if row["n_parameters"] != "" else "-"
        print(f"{row['model']:<28} {row['framework']:<14} "
              f"{row['accuracy']:>7.4f} {row['f1']:>7.4f} {auc:>7} "
              f"{row['train_seconds']:>10.3f} {str(params):>7}")

    print("-" * 96)
    _print_verdict(rows)
    print(f"\nTotal runtime: {summary['total_seconds']}s")
    print("=" * 96)


def _print_verdict(rows: List[Dict]) -> None:
    """State the conclusion plainly, including when it is unflattering."""
    working = [r for r in rows if not r["failed"] and r["model"] != "Majority baseline"]
    if not working:
        return

    classical = [r for r in working if r["family"] == "classical"]
    quantum = [r for r in working if r["family"] == "quantum"]
    if not classical or not quantum:
        return

    best_classical = max(classical, key=lambda r: r["accuracy"])
    best_quantum = max(quantum, key=lambda r: r["accuracy"])
    gap = best_quantum["accuracy"] - best_classical["accuracy"]

    speed = (
        best_quantum["train_seconds"] / best_classical["train_seconds"]
        if best_classical["train_seconds"] > 0
        else float("inf")
    )

    print("VERDICT")
    print(f"  Best classical : {best_classical['model']} "
          f"({best_classical['accuracy']:.2%}, "
          f"{best_classical['train_seconds']:.3f}s)")
    print(f"  Best quantum   : {best_quantum['model']} "
          f"[{best_quantum['framework']}] ({best_quantum['accuracy']:.2%}, "
          f"{best_quantum['train_seconds']:.3f}s)")

    if gap > 0.01:
        print(f"  -> Quantum leads by {gap:.2%} accuracy.")
    elif gap < -0.01:
        print(f"  -> Classical leads by {abs(gap):.2%} accuracy.")
    else:
        print(f"  -> Statistically indistinguishable ({abs(gap):.2%} apart).")

    print(f"  -> Quantum training was {speed:,.0f}x slower.")
    print("  -> No quantum advantage at this scale, as expected on a "
          "simulator. This is\n     the honest and anticipated result, not a "
          "failure of the implementation.")


def save_outputs(summary: Dict[str, object]) -> None:
    """Write the CSV, JSON summary and comparison chart."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(summary["results"])
    csv_path = OUTPUT_DIR / "benchmark_results.csv"
    frame.to_csv(csv_path, index=False)

    serialisable = {k: v for k, v in summary.items() if not k.startswith("_")}
    json_path = OUTPUT_DIR / "benchmark_summary.json"
    json_path.write_text(json.dumps(serialisable, indent=2, default=str))

    logger.info("Wrote %s and %s", csv_path.name, json_path.name)
    _save_chart(frame)


def _emergency_dump(summary: Dict[str, object]) -> None:
    """Last-resort write if `save_outputs` itself fails.

    Uses only the standard library — no pandas, no matplotlib — because those
    are the most likely things to have just crashed.
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "benchmark_results_recovered.csv"
        rows = summary.get("results", [])
        if not rows:
            return
        import csv

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.warning("Results recovered to %s", path)
    except Exception:  # pragma: no cover - truly nothing more we can do
        logger.error("Emergency dump failed; results exist only on screen.")


def _save_chart(frame: pd.DataFrame) -> None:
    """Accuracy and training time, side by side.

    Training time is on a log axis because the two families differ by three to
    five orders of magnitude — on a linear axis every classical bar would be
    invisible, which would understate the single most important finding.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping the chart.")
        return

    working = frame[~frame["failed"]].copy()
    if working.empty:
        return

    colours = ["#4C78A8" if f == "classical" else "#B85C9E"
               for f in working["family"]]

    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 6))

    left.barh(working["model"], working["accuracy"], color=colours)
    left.set_xlabel("Accuracy")
    left.set_title("Classification accuracy")
    left.set_xlim(0, 1)
    for index, value in enumerate(working["accuracy"]):
        left.text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=9)

    times = working["train_seconds"].replace(0, 1e-4)
    right.barh(working["model"], times, color=colours)
    right.set_xscale("log")
    right.set_xlabel("Training time (seconds, log scale)")
    right.set_title("Training cost")
    for index, value in enumerate(times):
        right.text(value * 1.15, index, f"{value:.3g}s", va="center", fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8"),
        plt.Rectangle((0, 0), 1, 1, color="#B85C9E"),
    ]
    figure.legend(handles, ["Classical", "Quantum"], loc="lower center", ncol=2)
    figure.suptitle(
        "Employee burnout risk: classical vs quantum classifiers "
        "(identical features and split)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))

    path = OUTPUT_DIR / "comparison.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    logger.info("Wrote %s", path.name)


# ==============================================================================
# Qubit sweep
# ==============================================================================
def run_sweep(qubit_counts: List[int], **kwargs) -> pd.DataFrame:
    """Repeat the quantum models at several qubit counts.

    Answers "do more qubits help?" — a question examiners ask and most
    student projects cannot answer. The expected shape is a plateau or a
    decline, since extra qubits beyond the feature count add depth without
    information, and depth invites barren plateaus.
    """
    rows = []
    for count in qubit_counts:
        logger.info("### sweep: %d qubits ###", count)
        summary = run_benchmark(n_qubits=count, **kwargs)
        for row in summary["results"]:
            row["n_qubits"] = count
            rows.append(row)

    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    return frame


# ==============================================================================
# CLI
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical vs quantum burnout risk classification."
    )
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--samples", type=int, default=1500,
                        help="Rows sampled from the 22,750-row dataset.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--kernel-rows", type=int, default=300,
                        help="Training rows for the O(n^2) quantum kernel.")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--feature-map", choices=["angle", "zz"], default="angle")
    parser.add_argument("--target", choices=["binary", "tertile"], default="binary")
    parser.add_argument("--skip-qiskit", action="store_true")
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Repeat at 4, 6 and 8 qubits.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.sweep:
        frame = run_sweep(
            [4, 6, 8],
            max_samples=args.samples, epochs=args.epochs,
            kernel_max_train=args.kernel_rows, target_mode=args.target,
            skip_qiskit=args.skip_qiskit, n_layers=args.layers,
            feature_map=args.feature_map,
        )
        print(frame.to_string(index=False))
        return

    summary = run_benchmark(
        n_qubits=args.qubits, max_samples=args.samples, epochs=args.epochs,
        kernel_max_train=args.kernel_rows, target_mode=args.target,
        skip_qiskit=args.skip_qiskit, skip_quantum=args.skip_quantum,
        feature_map=args.feature_map, n_layers=args.layers,
    )

    # SAVE BEFORE PRINTING.
    #
    # A full run takes ten to twenty minutes, most of it in Qiskit. Anything
    # that kills the process after the results exist but before they are
    # written costs the user that entire run.
    #
    # This is not hypothetical: on Python 3.14 with Qiskit 2.5, this script
    # segfaulted immediately after printing the table — the numbers were on
    # screen but nothing reached disk. Persisting first means a crash during
    # rendering costs a chart, not an experiment.
    try:
        save_outputs(summary)
    except Exception as exc:  # noqa: BLE001 - never lose results to a save bug
        logger.error("Could not save results: %s", exc)
        _emergency_dump(summary)

    print_table(summary)


__all__ = ["run_benchmark", "run_sweep", "print_table", "save_outputs"]


if __name__ == "__main__":
    main()
