"""Public entry point: ordinary training, prediction and public-label evaluation."""

import argparse
import json
import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
from threadpoolctl import threadpool_limits
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from flow_classifier import CLASSES, TASKS, task1, task2

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "flow_id",
    "split",
    "transport",
    "start_rel",
    "duration",
    "server_port",
    "packet_count_up",
    "packet_count_down",
    "bytes_up",
    "bytes_down",
    "mean_pkt_size_up",
    "mean_pkt_size_down",
    "std_pkt_size_up",
    "std_pkt_size_down",
    "mean_iat",
    "std_iat",
    "min_iat",
    "max_iat",
    "burst_count",
    "up_down_packet_ratio",
    "up_down_byte_ratio",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def task_names(selection):
    return TASKS if selection == "both" else (selection,)


def features(path, task):
    raw = pd.read_csv(path)
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if raw.empty or raw.flow_id.isna().any() or (not raw.flow_id.is_unique):
        raise ValueError(f"{path}: flow_id must be nonempty, unique and nonmissing")
    if not raw.transport.isin(("tcp", "udp")).all():
        raise ValueError(f"{path}: transport must contain tcp/udp strings")
    if not np.isfinite(raw.start_rel.to_numpy(float)).all():
        raise ValueError(f"{path}: start_rel must be finite")
    if task == TASKS[1] and (not (raw.packet_count_up + raw.packet_count_down).eq(5).all()):
        raise ValueError(f"{path}: Task 2 requires exactly five packets per flow")
    return raw.sort_values("flow_id", kind="stable").reset_index(drop=True)


def labels(path, raw):
    y = pd.read_csv(path)
    if not {"flow_id", "app"} <= set(y.columns):
        raise ValueError(f"{path}: labels require flow_id and app")
    if not y.flow_id.is_unique or set(y.flow_id) != set(raw.flow_id):
        raise ValueError(f"{path}: label IDs must match feature IDs exactly")
    answer = y.set_index("flow_id").loc[raw.flow_id, "app"].to_numpy()
    if not set(answer) <= set(CLASSES):
        raise ValueError(f"{path}: unknown or missing class labels")
    return answer


def aligned(full, prefix):
    if full.flow_id.tolist() != prefix.flow_id.tolist():
        raise ValueError("Task 1 requires aligned Task 2 prefix features for the same IDs")
    if not np.array_equal(full.transport, prefix.transport):
        raise ValueError("Full-flow/prefix transport values disagree")
    if not np.allclose(full.start_rel, prefix.start_rel, rtol=0, atol=1e-09):
        raise ValueError("Full-flow/prefix start_rel values disagree")


def training_data(data_dir, task, split):
    frames, targets = ([], [])
    for name in ("train",) if split == "train" else ("train", "validation"):
        folder = Path(data_dir) / task
        raw = features(folder / f"{name}_features.csv", task)
        targets.extend(labels(folder / f"{name}_labels.csv", raw))
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    if not raw.flow_id.is_unique:
        raise ValueError("Training splits contain overlapping IDs")
    order = np.argsort(raw.flow_id.to_numpy(), kind="stable")
    return (raw.iloc[order].reset_index(drop=True), np.asarray(targets)[order])


def plan_for(raw, y, task, scope, fold_plan=None):
    if set(y) != set(CLASSES) or min((np.sum(y == c) for c in CLASSES)) < 3:
        raise ValueError("Training requires all six classes, with at least three rows each")
    if fold_plan:
        plan = load_fold_plan(fold_plan)[f"{task}/{scope}"]
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=401)
        plan = [
            dict(fit_ids=raw.flow_id.iloc[fi].tolist(), held_ids=raw.flow_id.iloc[he].tolist())
            for fi, he in cv.split(raw, y)
        ]
    seen = []
    all_ids = set(raw.flow_id)
    by_id = dict(zip(raw.flow_id, y, strict=True))
    for entry in plan:
        fit, held = (entry["fit_ids"], entry["held_ids"])
        if (
            len(set(fit)) != len(fit)
            or len(set(held)) != len(held)
            or set(fit) & set(held)
            or (set(fit) | set(held) != all_ids)
        ):
            raise ValueError("Invalid fold plan: each fold must partition all training IDs")
        if {by_id[f] for f in fit} != set(CLASSES):
            raise ValueError("An inner training fold is missing a class")
        seen.extend(held)
    if len(plan) != 3 or len(seen) != len(set(seen)) or set(seen) != all_ids:
        raise ValueError("Three inner folds must hold out every training row exactly once")
    return plan


def load_recipes():
    """Construct unfitted learners from JSON, with an explicit type whitelist.

    No fitted artifact, arbitrary module import, or executable expression is
    needed to construct the training configuration.
    """
    from catboost import CatBoostClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, QuantileTransformer
    from flow_classifier import features as feature_ops

    constructors = {
        cls.__name__: cls
        for cls in (
            CatBoostClassifier,
            ColumnTransformer,
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            SimpleImputer,
            Pipeline,
            FunctionTransformer,
            OneHotEncoder,
            QuantileTransformer,
            feature_ops.BinaryTransportEncoder,
        )
    }
    functions = {name: getattr(feature_ops, name) for name in ("signed_log", "signed_log1p")}

    def restore(value):
        if isinstance(value, list):
            return [restore(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$estimator" in value:
            if value["$estimator"] not in constructors:
                raise ValueError("Unsupported configured estimator")
            return constructors[value["$estimator"]](**restore(value["params"]))
        if "$function" in value:
            if value["$function"] not in functions:
                raise ValueError("Unsupported configured transformation")
            return functions[value["$function"]]
        if "$tuple" in value:
            return tuple(restore(item) for item in value["$tuple"])
        if "$dtype" in value:
            return np.dtype(value["$dtype"]).type
        if "$array" in value:
            return np.asarray(value["$array"], dtype=value["dtype"])
        return {key: restore(item) for key, item in value.items()}

    config = json.loads((ROOT / "config/fitting.json").read_text())
    if config["format"] != 2:
        raise ValueError("Unsupported fitting configuration version")
    return restore(config["recipes"])


def train(data_dir, output_dir, split="train-validation", selection="both", fold_plan=None):
    """Fit from CSV; saved models/verification outputs are never training inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scope = "train247" if split == "train" else "full330"
    recipes = load_recipes()
    records = []
    for task in task_names(selection):
        raw, y = training_data(data_dir, task, split)
        prefix = None
        if task == TASKS[0]:
            prefix, py = training_data(data_dir, TASKS[1], split)
            aligned(raw, prefix)
            if not np.array_equal(y, py):
                raise ValueError("Task 1/Task 2 public training labels disagree")
        recipe = dict(recipes[task])
        plan = plan_for(raw, y, task, scope, fold_plan)
        recipe["memberships"] = {scope: plan}
        before = perf_counter()
        model = (
            task1.fit(recipe, raw, prefix, y, scope) if task == TASKS[0] else task2.fit(recipe, raw, y, scope)
        )
        seconds = perf_counter() - before
        joblib.dump(model, output_dir / f"{task}.joblib", compress=3)
        write_json(output_dir / f"{task}_folds.json", {f"{task}/{scope}": plan})
        records.append(
            dict(
                task=task,
                split=split,
                training_rows=len(raw),
                fit_seconds=seconds,
                model=f"{task}.joblib",
                class_counts={c: int(np.sum(y == c)) for c in CLASSES},
            )
        )
    report = dict(
        completed=True, records=records, fold_plan="explicit" if fold_plan else "stratified_3fold_seed401"
    )
    write_json(output_dir / "training_summary.json", report)
    return report


def predict_files(model_dir, output, task1_csv=None, task2_csv=None, selection="both"):
    """Predict any compatible CSV, without labels, manifests or expected outputs."""
    if not task1_csv and (not task2_csv):
        raise ValueError("Supply --task1-features and/or --task2-features")
    if task1_csv and (not task2_csv):
        raise ValueError("Task 1 also requires --task2-features for the aligned prefixes")
    if selection == TASKS[0] and (not task1_csv):
        raise ValueError("Task 1 requires --task1-features and aligned --task2-features")
    requested = task_names(selection)
    raw2 = features(task2_csv, TASKS[1]) if task2_csv else None
    raw1 = features(task1_csv, TASKS[0]) if task1_csv and TASKS[0] in requested else None
    if raw1 is not None:
        aligned(raw1, raw2)
    submission, records = ({}, [])
    for task, raw in ((TASKS[0], raw1), (TASKS[1], raw2)):
        if raw is None or task not in requested:
            continue
        before = perf_counter()
        model = joblib.load(Path(model_dir) / f"{task}.joblib")
        load_seconds = perf_counter() - before
        if set(model["fit_ids"]) & set(raw.flow_id):
            raise ValueError(f"{task}: query IDs overlap training IDs; use held-out/new flows")
        before = perf_counter()
        answer = predict_model(task, model, raw, raw2)
        records.append(
            dict(
                task=task,
                rows=len(raw),
                model_load_seconds=load_seconds,
                prediction_seconds=perf_counter() - before,
            )
        )
        submission[task] = dict(zip(raw.flow_id, answer.tolist(), strict=True))
    write_json(output, submission)
    return dict(completed=True, records=records)


def evaluate(predictions, labels_dir):
    submission = json.loads(Path(predictions).read_text())
    report = {}
    for task, values in submission.items():
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        raw = pd.DataFrame({"flow_id": list(values)})
        y = labels(Path(labels_dir) / task / "validation_labels.csv", raw)
        p = np.asarray(list(values.values()))
        if not set(p) <= set(CLASSES):
            raise ValueError("Unknown predicted class")
        report[task] = dict(
            rows=len(y),
            correct=int(np.sum(y == p)),
            accuracy=float(accuracy_score(y, p)),
            macro_f1=float(f1_score(y, p, labels=CLASSES, average="macro", zero_division=0)),
        )
    return report


def load_fold_plan(path):
    value = json.loads(Path(path).read_text())
    return value.get("memberships", value)


def predict_model(task, model, raw, prefix=None):
    return task1.predict(model, raw, prefix) if task == TASKS[0] else task2.predict(model, raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train", help="Cold-fit from labeled CSV files")
    fitting.add_argument("--data-dir", required=True)
    fitting.add_argument("--output-dir", required=True)
    fitting.add_argument("--split", choices=("train", "train-validation"), default="train-validation")
    fitting.add_argument("--task", choices=("both", *TASKS), default="both")
    fitting.add_argument("--fold-plan", help="Optional explicit three-fold membership JSON")
    inference = commands.add_parser("predict", help="Predict CSV files with saved models")
    inference.add_argument("--model-dir", required=True)
    inference.add_argument("--task", choices=("both", *TASKS), default="both")
    inference.add_argument("--task1-features")
    inference.add_argument("--task2-features")
    inference.add_argument("--output", required=True)
    scoring = commands.add_parser("evaluate", help="Score a prediction JSON against public validation labels")
    scoring.add_argument("--predictions", required=True)
    scoring.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "train":
            result = train(args.data_dir, args.output_dir, args.split, args.task, args.fold_plan)
        elif args.command == "predict":
            result = predict_files(
                args.model_dir, args.output, args.task1_features, args.task2_features, args.task
            )
        else:
            result = evaluate(args.predictions, args.data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
