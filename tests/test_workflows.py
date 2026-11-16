"""Public-interface regressions; no competition CSVs or private labels required."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flow_classifier import CLASSES
from flow_classifier import features as feature_ops, kernel, task1, support
from run import TASKS, features, plan_for, load_recipes


class WorkflowContracts(unittest.TestCase):
    @staticmethod
    def small_fitting_rows():
        gaps = np.array([0.0, 0.0, 0.3, 1.0])
        rows = []
        for i in range(len(CLASSES)):
            rows.append(
                dict(
                    flow_id=f"synthetic_{i}",
                    split="train",
                    transport="tcp",
                    start_rel=float(i),
                    duration=float(gaps.sum()),
                    server_port=443,
                    packet_count_up=2,
                    packet_count_down=3,
                    bytes_up=290,
                    bytes_down=222,
                    mean_pkt_size_up=145.0,
                    mean_pkt_size_down=74.0,
                    std_pkt_size_up=0.0,
                    std_pkt_size_down=0.0,
                    mean_iat=float(gaps.mean()),
                    std_iat=float(gaps.std()),
                    min_iat=float(gaps.min()),
                    max_iat=float(gaps.max()),
                    burst_count=2,
                    up_down_packet_ratio=2 / 3,
                    up_down_byte_ratio=290 / 222,
                )
            )
        return pd.DataFrame(rows)

    def test_removed_experts_are_not_fitted(self):
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

        raw = self.small_fitting_rows()
        recipe = load_recipes()[TASKS[0]]
        with patch.object(
            task1, "fresh_component", side_effect=lambda spec, frame, y: {"kind": spec["kind"]}
        ) as fresh, patch.object(
            ExtraTreesClassifier, "fit", side_effect=AssertionError("Unexpected whole ET fit")
        ), patch.object(
            HistGradientBoostingClassifier, "fit", return_value="HGB"
        ) as hgb, patch.object(
            kernel, "fit_kernel", return_value={"fixed": True}
        ):
            primary = task1.fit_primary(recipe, raw, raw.copy(), np.array(CLASSES))
        self.assertEqual(fresh.call_count, 2)
        self.assertEqual(hgb.call_count, 1)
        self.assertEqual(set(primary["baseline"]["models"]), {"compact", "full_cat", "hgb"})
        self.assertNotIn("whole", recipe["models"])
        self.assertNotIn("prefix_cat", recipe["models"])

    def test_kernel_uses_fixed_equal_weights_without_SVM_fit(self):
        from sklearn.svm import SVC

        raw = self.small_fitting_rows()
        frame = feature_ops.make_selected_frame(TASKS[0], raw, raw.copy())
        central = load_recipes()[TASKS[0]]["models"]["compact"]
        structured, _ = feature_ops.structural(frame, feature_ops.size_ops.structural_catalog(raw))
        with patch.object(SVC, "fit", side_effect=AssertionError("SVM fitting is forbidden")):
            fitted = kernel.fit_kernel(
                raw, structured, np.array(CLASSES), central["preprocess"], central["columns"]
            )
        np.testing.assert_allclose(fitted["fit_gram"], np.full((6, 6), 0.8), atol=0, rtol=0)
        self.assertNotIn("learned", fitted)

    def test_protocol_guard_counts_only_fitting_examples(self):
        raw = pd.DataFrame(dict(flow_id=["a", "b", "c"], transport=["tcp", "tcp", "udp"]))
        guard = support.fit_protocol_guard(raw, np.array([CLASSES[0], CLASSES[0], CLASSES[1]]), 2)
        self.assertEqual(guard["fit_ids"], ["a", "b", "c"])
        self.assertEqual(guard["minimum_rows"], 2)
        np.testing.assert_array_equal(guard["counts"][:2], [[2, 0], [0, 1]])
        self.assertEqual(guard["counts"].sum(), 3)

    def test_protocol_guard_treats_every_class_and_protocol_equally(self):
        for protocol, code in (("tcp", 0), ("udp", 1)):
            raw = pd.DataFrame(dict(transport=[protocol]))
            for j, old in enumerate(CLASSES):
                new = CLASSES[(j + 1) % len(CLASSES)]
                counts = np.full((len(CLASSES), 2), 2)
                guard = dict(classes=CLASSES, counts=counts, minimum_rows=2)
                self.assertEqual(support.limit_protocol_changes(guard, raw, [old], [new])[0], new)
                for count in (0, 1):
                    counts[j, code] = count
                    self.assertEqual(support.limit_protocol_changes(guard, raw, [old], [new])[0], old)
                    self.assertEqual(support.limit_protocol_changes(guard, raw, [new], [old])[0], new)
                counts[j, code] = 2
                # Scarcity in the other protocol must not veto this query.
                counts[j, 1 - code] = 0
                self.assertEqual(support.limit_protocol_changes(guard, raw, [old], [new])[0], new)

    def test_task2_raw_statistics_match_task1_prefix_statistics(self):
        recipe = load_recipes()[TASKS[1]]
        chosen = recipe["columns"]
        size_columns = set(feature_ops.EXTREMA_COLUMNS)
        fingerprints = {"fp_up_bytes", "fp_up_max_bound"}
        raw = set(chosen) - size_columns - fingerprints - {"transport"}
        expected = {c[2:] for c in feature_ops.PREFIX_COLUMNS}
        self.assertEqual(raw, expected)
        self.assertEqual(len(raw), 16)
        self.assertIn("transport", chosen)
        self.assertEqual(len(chosen), 33)
        self.assertNotIn("rotation_columns", recipe)

    def test_shared_prefix_bounds_handle_absent_single_and_two_packet_directions(self):
        raw = pd.DataFrame(
            dict(
                packet_count_up=[0, 1, 2],
                mean_pkt_size_up=[0.0, 66.0, 100.0],
                std_pkt_size_up=[0.0, 0.0, 20.0],
                packet_count_down=[2, 2, 2],
                mean_pkt_size_down=[100.0] * 3,
                std_pkt_size_down=[20.0] * 3,
            )
        )
        bounds = feature_ops.prefix_size_bounds(raw)
        self.assertTrue(bounds.loc[0, "p_up_exact_min":"p_up_exact_max"].isna().all())
        self.assertEqual(bounds.loc[0, "p_up_exact_extrema_available"], 0)
        np.testing.assert_array_equal(bounds.loc[1, ["p_up_exact_min", "p_up_exact_max"]], [66, 66])
        np.testing.assert_array_equal(bounds.loc[2, ["p_up_exact_min", "p_up_exact_max"]], [80, 120])
        for low in ("min_lower", "min_upper"):
            self.assertEqual(bounds.loc[2, "p_up_" + low], 80)
        for high in ("max_lower", "max_upper"):
            self.assertEqual(bounds.loc[2, "p_up_" + high], 120)

    def test_shared_size_and_fingerprint_input_profiles(self):
        recipes = load_recipes()
        a = recipes[TASKS[0]]["models"]["compact"]["columns"]
        b = recipes[TASKS[1]]["columns"]
        self.assertEqual(
            [c for c in a if c in feature_ops.EXTREMA_COLUMNS],
            [c for c in b if c in feature_ops.EXTREMA_COLUMNS],
        )
        self.assertEqual([c for c in a if c.startswith("fp_")], [c for c in b if c.startswith("fp_")])

    def test_time_filter_ignores_future_and_simultaneous_labels(self):
        prior = np.full(6, 1 / 6)
        first = np.array([1.0, 0, 0, 0, 0, 0])
        model = dict(
            origin=0.0,
            times=np.array([1.0, 2.0, 3.0]),
            prior=prior,
            tau=1.0,
            forward=np.array([first, np.roll(first, 1), np.roll(first, 2)]),
            query_emission=np.stack([np.eye(6), np.eye(6)]),
        )
        query = pd.DataFrame(dict(start_rel=[0.0, 1.0, 1.5, 2.0], transport=["tcp"] * 4))
        original = support.time_context(model, query)
        np.testing.assert_allclose(original[:2], [prior, prior])
        np.testing.assert_allclose(original[3], np.exp(-1) * first + -np.expm1(-1) * prior)
        model["forward"][1:] = 1 / 6
        np.testing.assert_array_equal(original, support.time_context(model, query))

    def test_capture_prefix_selection_never_uses_later_or_tied_rows(self):
        raw = pd.DataFrame(dict(flow_id=["a", "b", "c"], start_rel=[1.0, 2.0, 2.0]))

        def expert(frame, labels, cache, *, keys):
            self.assertEqual(keys, frame.flow_id.tolist())
            return dict(fit_ids=frame.flow_id.tolist())

        with patch.object(support, "size_signatures", return_value=raw.flow_id.tolist()), patch.object(
            support, "fit_capture_expert", expert
        ):
            model = support.fit_past_capture(
                raw,
                np.array(CLASSES[:3]),
            )
        query = pd.DataFrame(dict(start_rel=[0.0, 1.0, 1.5, 2.0, 3.0]))
        labels = np.array([CLASSES[-1]] * 5)
        seen = []

        def select(snapshot, rows, reference):
            seen.append((snapshot["experts"]["strict"]["fit_ids"], rows.start_rel.tolist()))
            return np.array([CLASSES[len(snapshot["experts"]["strict"]["fit_ids"]) - 1]] * len(rows))

        with patch.object(support, "select_capture", select), patch.object(
            support, "fit_capture_expert", side_effect=AssertionError("no inference fit")
        ):
            predicted = support.select_past_capture(
                model, query, labels
            )
        self.assertEqual(seen, [(["a"], [1.5, 2.0]), (["a", "b", "c"], [3.0])])
        np.testing.assert_array_equal(predicted[:2], labels[:2])

    def test_joined_candidates_preserve_flow_and_class_order(self):
        # OOF folds can be unequal: class-major concatenation silently associates
        # a candidate with the wrong flow when the ranker expects flow-major rows.
        blocks = []
        for ids in ([8, 2], [4]):
            blocks.append(
                dict(
                    x=np.array([[[i, c] for c in range(6)] for i in ids]),
                    default=np.array(ids),
                    votes=np.zeros((len(ids), 6)),
                )
            )
        joined = task1.join_candidate_rows(blocks)
        np.testing.assert_array_equal(joined["x"][:, :, 0], [[8] * 6, [2] * 6, [4] * 6])
        np.testing.assert_array_equal(joined["x"][:, :, 1], [list(range(6))] * 3)
        with self.assertRaises(ValueError):
            task1.fit_candidate_ranker(joined, [CLASSES[0]])

    def test_missing_fingerprints_do_not_create_similarity(self):
        fit = pd.DataFrame({"key": ["tcp|NA", "tcp|145"]})
        query = pd.DataFrame({"key": ["tcp|NA", "tcp|145", "tcp|159"]})
        np.testing.assert_array_equal(kernel.categorical_component(fit, query), [[0, 0], [0, 1], [0, 0]])
        # Direct callers retain physical-input validation, independently of CLI.
        with self.assertRaises(ValueError):
            feature_ops.reconstruct(pd.DataFrame({"packet_count_up": [2], "packet_count_down": [2]}))

    def test_readable_recipes_construct_only_unfitted_estimators(self):
        recipes = load_recipes()
        for spec in recipes[TASKS[0]]["models"].values():
            estimator = spec["model"] if isinstance(spec, dict) else spec.named_steps["model"]
            if hasattr(estimator, "is_fitted"):
                self.assertFalse(estimator.is_fitted())
            else:
                self.assertFalse(hasattr(estimator, "classes_"))
        self.assertFalse(recipes[TASKS[1]]["catboost"].is_fitted())

    def test_final_task2_has_only_original_coordinate_et(self):
        model = joblib.load(ROOT / "models" / f"{TASKS[1]}.joblib")
        self.assertNotIn("pca", model)
        self.assertNotIn("rotation", model["source"])
        forest = model["source"]["base"]["pipeline"].named_steps["model"]
        self.assertEqual(len(forest.estimators_), 300)

    def test_generated_folds_cover_once_and_keep_all_classes(self):
        y = np.repeat(CLASSES, 3)
        raw = pd.DataFrame({"flow_id": [f"id_{i}" for i in range(len(y))]})
        plan = plan_for(raw, y, TASKS[0], "full330")
        held = [f for fold in plan for f in fold["held_ids"]]
        self.assertEqual(set(held), set(raw.flow_id))
        self.assertEqual(len(held), len(set(held)))
        for fold in plan:
            self.assertFalse(set(fold["fit_ids"]) & set(fold["held_ids"]))

    def test_bad_fold_plan_is_rejected_before_fitting(self):
        y = np.repeat(CLASSES, 3)
        raw = pd.DataFrame({"flow_id": [f"id_{i}" for i in range(len(y))]})
        plan = plan_for(raw, y, TASKS[0], "full330")
        plan[0]["held_ids"].append(plan[0]["held_ids"][0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({f"{TASKS[0]}/full330": plan}))
            with self.assertRaises(ValueError):
                plan_for(raw, y, TASKS[0], "full330", path)

    def test_missing_minority_class_is_rejected(self):
        y = np.repeat(CLASSES[:-1], 3)
        raw = pd.DataFrame({"flow_id": [f"id_{i}" for i in range(len(y))]})
        with self.assertRaises(ValueError):
            plan_for(raw, y, TASKS[0], "full330")

    def test_inference_without_config_fixtures_labels_or_original_workspace(self):
        # Copy only what inference requires; unrelated paths and historical IDs
        # are absent. The single new row is physically consistent, not a fixture.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                ROOT / "flow_classifier",
                root / "flow_classifier",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(ROOT / "models", root / "models")
            shutil.copy2(ROOT / "run.py", root / "run.py")
            gaps = np.array([0.0, 0.0, 0.3, 1.0])
            row = dict(
                flow_id="new_synthetic_flow",
                split="query",
                transport="tcp",
                start_rel=10.0,
                duration=float(gaps.sum()),
                server_port=443,
                packet_count_up=2,
                packet_count_down=3,
                bytes_up=290,
                bytes_down=222,
                mean_pkt_size_up=145.0,
                mean_pkt_size_down=74.0,
                std_pkt_size_up=0.0,
                std_pkt_size_down=0.0,
                mean_iat=float(gaps.mean()),
                std_iat=float(gaps.std()),
                min_iat=float(gaps.min()),
                max_iat=float(gaps.max()),
                burst_count=2,
                up_down_packet_ratio=2 / 3,
                up_down_byte_ratio=290 / 222,
            )
            path = root / "query.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            features(path, TASKS[1])
            env = dict(os.environ, PYTHONPATH="", PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning")
            base = [
                sys.executable,
                str(root / "run.py"),
                "predict",
                "--model-dir",
                str(root / "models"),
                "--task2-features",
                str(path),
            ]
            subprocess.run(
                [*base, "--task1-features", str(path), "--output", str(root / "both.json")],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [*base, "--output", str(root / "task2.json")],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            both = json.loads((root / "both.json").read_text())
            only = json.loads((root / "task2.json").read_text())
            self.assertEqual(only, {TASKS[1]: both[TASKS[1]]})
            self.assertEqual(set(both), set(TASKS))
            self.assertTrue(all(x["new_synthetic_flow"] in CLASSES for x in both.values()))
            # A Task 1-only fit must be usable without a Task 2 model file.
            (root / "models" / f"{TASKS[1]}.joblib").unlink()
            subprocess.run(
                [
                    *base,
                    "--task",
                    TASKS[0],
                    "--task1-features",
                    str(path),
                    "--output",
                    str(root / "task1.json"),
                ],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads((root / "task1.json").read_text()), {TASKS[0]: both[TASKS[0]]})


if __name__ == "__main__":
    unittest.main()
