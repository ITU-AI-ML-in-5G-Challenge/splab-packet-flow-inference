"""Capture routing and shared preprocessing contracts without competition labels."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flow_classifier import CLASSES, TASKS, support, task2, features as ops


class FakeRF:
    classes_ = np.asarray(CLASSES)

    def __init__(self, probability):
        self.probability = np.asarray(probability)
        self.calls = []

    def predict_proba(self, matrix):
        self.calls.append(len(matrix))
        return np.tile(self.probability, (len(matrix), 1))


class EfficiencyContracts(unittest.TestCase):
    def test_task2_retains_time_and_removes_standalone_gap_helper(self):
        model = joblib.load(ROOT / 'models' / f'{TASKS[1]}.joblib')
        self.assertIsNotNone(model['time'])
        self.assertNotIn('gap', model['source'])
        self.assertFalse(hasattr(support, 'fit_gap'))
        self.assertFalse(hasattr(support, 'predict_gap'))

    def test_training_never_constructs_fallback(self):
        raw = pd.DataFrame(dict(flow_id=['a', 'b'], start_rel=[1., 2.]))
        with patch.object(support, 'size_signatures', return_value=['a', 'b']) as signatures, patch.object(
            support, 'fit_capture_expert', return_value={'entries': {}}
        ) as fit:
            model = support.fit_past_capture(raw, np.array(['google', 'x']))
        signatures.assert_called_once_with(raw)
        self.assertEqual(fit.call_count, 2)
        self.assertTrue(all(set(s['experts']) == {'strict'} for s in model['snapshots']))
        self.assertTrue(all(set(s) == {'experts'} for s in model['snapshots']))

    def route(self, probability, reference, active=None):
        raw = pd.DataFrame(dict(row=range(len(reference))))
        key = ('tcp', 2, 200, 20000, 3, 300, 30000)
        forest = FakeRF(probability)
        entry = dict(learner=forest, active=list(CLASSES) if active is None else active, origin=0)
        model = dict(experts={'strict': dict(entries={key: entry})})
        with patch.object(support, 'size_signatures', lambda r: [key] * len(r)), patch.object(
            support, 'capture_features', lambda r, origin: np.zeros((len(r), 1))
        ):
            answer = support.select_capture(model, raw, np.asarray(reference))
        return answer, forest

    def test_RF_applies_to_every_eligible_reference_class(self):
        probability = [.9, .02, .02, .02, .02, .02]
        answer, forest = self.route(probability, ['facebook', 'google'])
        np.testing.assert_array_equal(answer, ['facebook', 'facebook'])
        self.assertEqual(forest.calls, [2])

    def test_same_label_prediction_is_still_evaluated(self):
        probability = [.8, .04, .04, .04, .04, .04]
        answer, forest = self.route(probability, ['facebook'])
        np.testing.assert_array_equal(answer, ['facebook'])
        self.assertEqual(forest.calls, [1])

    def test_RF_accepts_candidate_below_former_confidence_threshold(self):
        probability = [.042, .79, .042, .042, .042, .042]
        answer, _ = self.route(probability, ['facebook'])
        np.testing.assert_array_equal(answer, ['google'])

    def test_inactive_reference_skips_forest(self):
        probability = [.9, .02, .02, .02, .02, .02]
        answer, forest = self.route(probability, ['x'], active=['facebook', 'google'])
        np.testing.assert_array_equal(answer, ['x'])
        self.assertEqual(forest.calls, [])

    def test_distributed_model_contains_only_strict_rf_with_shared_provenance(self):
        model = joblib.load(ROOT / 'models' / f'{TASKS[1]}.joblib')
        capture = model['source']['capture']
        snapshots = capture['snapshots']
        self.assertTrue(all(set(s) == {'experts'} for s in snapshots))
        self.assertTrue(all(set(s['experts']) == {'strict'} for s in snapshots))
        entries = [e for s in snapshots for e in s['experts']['strict']['entries'].values()]
        self.assertTrue(all(e['learner'].n_features_in_ == 9 for e in entries))
        self.assertTrue(all(set(e) == {'learner', 'active', 'origin', 'fit_ids'} for e in entries))
        self.assertEqual(len({id(e) for e in entries}), capture['distinct_forests'])

    def test_shared_frame_once_and_independent_preprocessors_still_work(self):
        model = joblib.load(ROOT / 'models' / f'{TASKS[1]}.joblib')
        source = model['source']
        gaps = np.array([0., .01, .04, .05])
        raw = pd.DataFrame([dict(
            flow_id='synthetic_query', split='test', transport='tcp', start_rel=1., server_port=443,
            duration=.1, packet_count_up=2, packet_count_down=3, bytes_up=211, bytes_down=222,
            mean_pkt_size_up=105.5, mean_pkt_size_down=74., std_pkt_size_up=39.5,
            std_pkt_size_down=0., mean_iat=.025, std_iat=gaps.std(), min_iat=0., max_iat=.05,
            burst_count=2, up_down_packet_ratio=2/3, up_down_byte_ratio=211/222,
        )])
        frame = ops.prefix_frame(raw)
        base = source['base']
        encoded = base['pipeline'][:-1].transform(frame[base['columns']])
        matrix, _ = ops.native(base['pipeline'].named_steps['preprocess'], frame, encoded)
        expected = (base['pipeline'].named_steps['model'].predict_proba(encoded),
                    base['catboost'].predict_proba(matrix), task2.size_probabilities(source['latent'], raw)[0])
        for separate in (False, True):
            current = dict(base)
            if separate:
                current['pipeline'] = copy.deepcopy(current['pipeline'])
            with patch.object(ops, 'prefix_frame', wraps=ops.prefix_frame) as build:
                actual = task2.source_probabilities(dict(base=current, latent=source['latent']), raw)
            self.assertEqual(build.call_count, 1)
            for a, b in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
