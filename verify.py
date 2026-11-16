"""Optional frozen-benchmark regression check; ordinary commands never import this."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from importlib.metadata import version
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier
from flow_classifier import features as feature_ops, sizes as size_ops
from flow_classifier import TASKS, support as support_ops
from run import predict_model as predict
from run import features, training_data, write_json, load_recipes

ROOT = Path(__file__).resolve().parent
BENCHMARK = json.loads((ROOT / "verification/benchmark.json").read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(*args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::FutureWarning",
            "-W",
            "ignore::UserWarning",
            str(ROOT / "run.py"),
            *map(str, args),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def feature_checks(data_dir):
    expected = BENCHMARK["selected_inputs"]
    recipes = load_recipes()
    fit_prefix, _ = training_data(data_dir, TASKS[1], "train-validation")
    catalog = size_ops.structural_catalog(fit_prefix)
    vocabulary = size_ops.vocabulary(catalog, np.arange(len(catalog)))
    checked = []
    for scope in ("full330", "test81"):
        if scope == "full330":
            raw, _ = training_data(data_dir, TASKS[0], "train-validation")
            prefix = fit_prefix
        else:
            raw = features(data_dir / TASKS[0] / "test_features.csv", TASKS[0])
            prefix = features(data_dir / TASKS[1] / "test_features.csv", TASKS[1])
        frame = feature_ops.make_selected_frame(TASKS[0], raw, prefix)
        body = feature_ops.compact_features(TASKS[0], raw, prefix)
        full = feature_ops.structural_query(frame, size_ops.structural_catalog(prefix), vocabulary)
        from flow_classifier import task1
        frames = dict(main61=frame[recipes[TASKS[0]]["models"]["compact"]["columns"]],
                      cat63=full[recipes[TASKS[0]]["models"]["full_cat"]["columns"]],
                      hgb17=body[recipes[TASKS[0]]["body_columns"]],
                      context190=pd.DataFrame(task1.context(frame, body)),
                      task2_33=feature_ops.prefix_frame(prefix)[recipes[TASKS[1]]["columns"]])
        for name, values in frames.items():
            key = scope + "/" + name
            header = json.dumps(dict(columns=list(values.columns), dtypes=[str(d) for d in values.dtypes]), sort_keys=True).encode()
            value = hashlib.sha256(header + pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes()).hexdigest()
            if value != expected[key]["sha256"] or list(values.shape) != expected[key]["shape"]:
                raise AssertionError(f"Feature fixture mismatch: {key}")
            checked.append(key)
    return checked


def invariant_checks(model_dir, data_dir, split):
    raw = {t: features(data_dir / t / f"{split}_features.csv", t) for t in TASKS}
    result = {}
    for task in TASKS:
        model = joblib.load(model_dir / f"{task}.joblib")
        a, b = raw[task], raw[TASKS[1]]
        fitted, truth = training_data(
            data_dir, task, "train" if split == "validation" else "train-validation"
        )
        assert model["fit_ids"] == fitted.flow_id.tolist()
        np.testing.assert_array_equal(model["fit_labels"], truth)
        temporal = model.get("time")
        if temporal is not None:
            assert temporal["fit_ids"] == fitted.flow_id.tolist()
            np.testing.assert_array_equal(temporal["fit_labels"], truth)
            assert temporal["reference_policy"] == "strictly_earlier_start"
            assert "backward" not in temporal
        assert set(model["fit_ids"]).isdisjoint(a.flow_id)
        before = time.perf_counter()
        with patch.object(
            RandomForestClassifier, "fit", side_effect=AssertionError("No fitting at inference")
        ):
            p = predict(task, model, a, b)
        seconds = time.perf_counter() - before
        np.testing.assert_array_equal(predict(task, model, a.iloc[::-1], b.iloc[::-1])[::-1], p)
        np.testing.assert_array_equal(predict(task, model, a.iloc[:1], b.iloc[:1]), p[:1])
        result[task] = dict(
            reverse_order=True,
            singleton_batch=True,
            disjoint_fit_ids=True,
            fitting_labels_match=True,
            prediction_seconds=seconds,
            no_RF_fitting_at_inference=True,
            no_backward_messages=True,
        )
        if task == TASKS[0]:
            recipe = load_recipes()[task]
            base = model["parent"]["base"]
            assert set(base["baseline"]["models"]) == {"compact", "full_cat", "hgb"}
            full_cat = base["baseline"]["models"].get("full_cat")
            if full_cat is not None:
                expected_trees = recipe["models"]["full_cat"]["model"].get_params()["iterations"]
                assert full_cat["model"].tree_count_ == expected_trees
                result[task]["full_cat_tree_count"] = expected_trees
            assert base["kernel"]["fit_categories"].shape[1] == 4
            result[task]["only_selected_models_fitted"] = True
            result[task]["fixed_kernel_no_SVM_model"] = True
            minimum = load_recipes()[task].get("minimum_protocol_rows", 0)
            if minimum:
                guard = model["protocol_guard"]
                expected = support_ops.fit_protocol_guard(fitted, truth, minimum)
                assert guard["minimum_rows"] == minimum
                assert guard["fit_ids"] == fitted.flow_id.tolist()
                np.testing.assert_array_equal(guard["counts"], expected["counts"])
                result[task]["protocol_guard_matches_fitting_rows"] = True
        if task == TASKS[1]:
            assert temporal is not None
            recipe = load_recipes()[task]
            assert "gap" not in recipe and "gap" not in model["source"]
            result[task]["temporal_correction_retained"] = True
            result[task]["standalone_gap_helper_not_fitted_or_stored"] = True
            assert "pca" not in model and "rotation" not in model["source"]
            forest = model["source"]["base"]["pipeline"].named_steps["model"]
            assert len(forest.estimators_) == 300
            result[task]["original_ET_tree_count"] = 300
            result[task]["no_rotated_ET_or_axes"] = True
            capture = model["source"]["capture"]
            assert "capture" not in recipe
            assert all(set(snapshot) == {"experts"} for snapshot in capture["snapshots"])
            assert all(set(snapshot["experts"]) == {"strict"} for snapshot in capture["snapshots"])
            assert all(entry["learner"].n_features_in_ == 9 for snapshot in capture["snapshots"]
                       for entry in snapshot["experts"]["strict"]["entries"].values())
            result[task]["RF_confidence_and_protection_settings_removed"] = True
            result[task]["fallback_RF_never_constructed_or_stored"] = True
            lookup = fitted.set_index("flow_id").start_rel
            signatures = support_ops.size_signatures(fitted)
            label_lookup = dict(zip(fitted.flow_id, truth))
            cuts = np.searchsorted(capture["times"], a.start_rel.to_numpy(float), side="left") - 1
            checked = set()
            for i, cut in enumerate(cuts):
                if cut < 0:
                    continue
                snapshot = capture["snapshots"][cut]
                for expert in snapshot["experts"].values():
                    for key, entry in expert["entries"].items():
                        assert (lookup.loc[entry["fit_ids"]] < a.start_rel.iloc[i]).all()
                        if cut in checked:
                            continue
                        prefix = fitted.start_rel.to_numpy() <= capture["times"][cut]
                        same_key = np.array([own == key for own in signatures]) & prefix
                        active = [c for c in model["source"]["support"]["classes"]
                                  if np.sum(same_key & (truth == c)) >= 3]
                        assert entry["active"] == active and len(active) >= 2
                        expected = [flow for flow, own, earlier in zip(fitted.flow_id, signatures, prefix)
                                    if earlier and support_ops.signature_profile(own) == support_ops.signature_profile(key)
                                    and label_lookup[flow] in active]
                        assert entry["fit_ids"] == expected
                checked.add(cut)
            result[task]["all_selected_capture_references_strictly_earlier"] = True
            result[task]["RF_support_and_training_pool_reconstructed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--inference-only", action="store_true", help="Check supplied models; skip cold refitting"
    )
    args = parser.parse_args()
    data_dir, out = args.data_dir.resolve(), args.output_dir.resolve()
    # Never overwrite the distributed final models during verification.
    if out == ROOT or out == ROOT / "models" or (ROOT / "models") in out.parents:
        raise ValueError("Choose a separate verification output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name, expected in BENCHMARK["data_sha256"].items():
        if sha(data_dir / name) != expected:
            raise AssertionError(f"Expected original competition CSV bytes: {name}")
    checked = feature_checks(data_dir)
    records = []
    plans = ROOT / "config/fitting.json"
    for split, train_split, fixture in (
        ("validation", "train", "validation_predictions.json"),
        ("test", "train-validation", "official_test_predictions.json"),
    ):
        if args.inference_only and split == "validation":
            continue
        models = ROOT / "models" if args.inference_only else out / split / "models"
        if not args.inference_only:
            command(
                "train",
                "--data-dir",
                data_dir,
                "--split",
                train_split,
                "--fold-plan",
                plans,
                "--output-dir",
                models,
            )
        output = out / split / "predictions.json"
        command(
            "predict",
            "--model-dir",
            models,
            "--task1-features",
            data_dir / TASKS[0] / f"{split}_features.csv",
            "--task2-features",
            data_dir / TASKS[1] / f"{split}_features.csv",
            "--output",
            output,
        )
        actual = json.loads(output.read_text())
        expected = BENCHMARK["predictions"][split]
        differences = {t: sum(actual[t].get(f) != c for f, c in expected[t].items()) for t in TASKS}
        if actual != expected:
            raise AssertionError(f"{split}: prediction mismatch: {differences}")
        if sha(output) != BENCHMARK["prediction_sha256"][split]:
            raise AssertionError(f"{split}: serialized JSON bytes differ")
        record = dict(
            split=split,
            rows_per_task={t: len(actual[t]) for t in TASKS},
            changed_predictions=differences,
            byte_identical=True,
            sha256=sha(output),
            invariance=invariant_checks(models, data_dir, split),
        )
        if not args.inference_only:
            record["training"] = json.loads((models / "training_summary.json").read_text())["records"]
        if split == "validation":
            from run import evaluate

            record["metrics"] = evaluate(output, data_dir)
        records.append(record)
    report = dict(
        passed=True,
        cold_retraining=not args.inference_only,
        feature_checks=checked,
        comparisons=records,
        total_seconds=time.perf_counter() - started,
        official_test_labels_available=False,
        official_scores="Previously reported scores; no new official scoring was performed.",
        environment=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            numerical_threads=1,
            packages={
                p: version(p)
                for p in ("numpy", "scipy", "pandas", "scikit-learn", "catboost", "joblib", "threadpoolctl")
            },
        ),
    )
    write_json(out / "verification_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
