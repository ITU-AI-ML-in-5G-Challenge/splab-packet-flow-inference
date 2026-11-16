# Application Inference from Packet Flows — SPlab

Solution for the [ITU AI/ML Application Inference from Packet Flows Challenge](https://competition.aiforgood.itu.int/web/challenges/challenge-page/495/overview), released by **SPlab** under the [MIT License](LICENSE).

This repository includes the selected Task 1 and Task 2 implementations, two trained models, explicit training folds and reproducibility checks. It predicts six applications: `facebook`, `google`, `instagram`, `mozilla`, `x` and `youtube`. No competition CSVs, external pretrained model or historical experiment code is included.

## Results

| Evaluation                                    | Task 1 Macro F1 | Task 1 Accuracy | Task 2 Macro F1 | Task 2 Accuracy |
| --------------------------------------------- | --------------: | --------------: | --------------: | --------------: |
| Public validation: train 247 → validation 83  |        0.881581 |        0.939759 |        0.852154 |        0.915663 |
| Official test: train+validation 330 → test 81 |    **0.971178** |    **0.962963** |    **0.951341** |    **0.938272** |

The reported official total is **96.125910 / 100**. These are organizer scores previously provided to the team. The supplied models reproduce that submission exactly; private test labels are unavailable here. Validation data and repeated official feedback informed development, so these results are not estimates from untouched model-selection holdouts. Inner out-of-fold (OOF) predictions train the integration layers; they are not independent full-method evaluation scores.

## Method at a glance

|                | Task 1: full-flow classification                                | Task 2: first-five-packet classification                                                           |
| -------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Required input | Full-flow CSV and aligned Task 2 prefix CSV                     | Task 2 prefix CSV only                                                                             |
| Content models | ExtraTrees, CatBoost, histogram gradient boosting, kernel ridge | ExtraTrees, CatBoost, latent packet-size density                                                   |
| Integration    | OOF-trained six-candidate ranker                                | OOF-trained log-probability fusion                                                                 |
| Final support  | Shared time correction and class/protocol support               | Shared time correction, class/protocol support, downstream-size interpolation and past-context RFs |

Size constraints use counts, byte sums, means and population standard deviations. They do not recover packet order or payload. TCP/UDP is encoded as 0/1. No IP addresses, DNS, SNI, payloads, metadata class-count assignment, application-pair override or query-flow-ID rule is used.

**Recording-context assumption.** Both tasks use `start_rel` and historical **training labels** from the same recording. Query-time messages and Task 2 context RFs reference strictly earlier labeled flows; equal/later starts are excluded. Global content learners and temporal parameters are trained using the entire given training range. Queries do not exchange labels or predictions. `flow_id` aligns rows and checks overlap; it is not a numerical predictor.

## Installation

Run the following commands from this repository's root. Python **3.12.14** on macOS arm64 is the verified reproduction environment. CPU execution uses one numerical thread; no GPU or external model service is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

`requirements.txt` lists the seven directly imported dependencies at the tested versions. `requirements-lock.txt` preserves the full dependency snapshot used for reproduction. Other Python/library versions and CPU architectures may change serialized-model compatibility or floating-point results.

## Obtain the data

Download the challenge data separately from the [official dataset and starter repository](https://github.com/FG-AI4H/application-inference-packet-flows-public). Follow the data provider's terms; this project's MIT license does not relicense that dataset.

```text
dataset/
├── task1_static/
│   ├── train_features.csv
│   ├── train_labels.csv
│   ├── validation_features.csv
│   ├── validation_labels.csv
│   └── test_features.csv
└── task2_realtime_prefix5/
    └── ... same filenames ...
```

```bash
export DATA_DIR=/absolute/path/to/dataset
```

## Predict with the supplied models

```bash
python run.py predict --model-dir models \
  --task1-features "$DATA_DIR/task1_static/test_features.csv" \
  --task2-features "$DATA_DIR/task2_realtime_prefix5/test_features.csv" \
  --output artifacts/test_predictions.json
```

The command prints model-loading and prediction times and writes:

```json
{
  "task1_static": { "flow_000007": "google" },
  "task2_realtime_prefix5": { "flow_000007": "youtube" }
}
```

For Task 2 alone, omit `--task1-features`. For Task 1 alone, supply both CSVs and add `--task task1_static`; only the Task 1 model is loaded.
Models retain training-derived references and labels used for time/support inference. Load `.joblib` files only from trusted sources: deserialization can execute Python code. Keep models with the corresponding source and pinned dependencies.

## Retrain the selected method

```bash
# Fit all 330 public labeled flows, including the inner OOF work.
python run.py train --data-dir "$DATA_DIR" --split train-validation \
  --fold-plan config/fitting.json --output-dir artifacts/final_models
```

Use `--model-dir artifacts/final_models` for inference after retraining.
For compatible new labeled data, omit `--fold-plan` to generate stratified three-fold memberships with seed 401. All six classes need at least three examples and every inner training fold must contain all classes. Exact competition reproduction uses the supplied fold plan. `--task task1_static` or `--task task2_realtime_prefix5` restricts fitting.

## Evaluate public validation data

The supplied final models already include the validation labels in their training scope. To evaluate the 83 validation rows without that overlap, first fit only the 247 training rows:

```bash
python run.py train --data-dir "$DATA_DIR" --split train \
  --fold-plan config/fitting.json --output-dir artifacts/train_only
python run.py predict --model-dir artifacts/train_only \
  --task1-features "$DATA_DIR/task1_static/validation_features.csv" \
  --task2-features "$DATA_DIR/task2_realtime_prefix5/validation_features.csv" \
  --output artifacts/validation_predictions.json
python run.py evaluate --data-dir "$DATA_DIR" \
  --predictions artifacts/validation_predictions.json
```

## Optional reproduction and tests

Normal training and prediction require only the commands above. The optional verifier calls the same `run.py train` / `run.py predict` commands and adds automatic comparison and invariant checks. Unit tests exercise additional cases such as invalid inputs and temporal-reference rules.

```bash
# Unit/standalone-interface checks: no competition CSVs required.
python -m unittest discover -s tests -v

# Cold refit at both 247/330-row scopes and compare original predictions.
python verify.py --data-dir "$DATA_DIR" --output-dir artifacts/reproduction

# Reload supplied models without refitting and verify test predictions.
python verify.py --data-dir "$DATA_DIR" \
  --output-dir artifacts/inference_check --inference-only
```

All **25 tests** pass locally. Cold refits preserve both validation/test JSONs byte for byte and ten frozen selected-input matrices. The winning test JSON SHA-256 is `9c20120c13898bb1faae4974bc0d1b52eee78ab05189a82ca1a11e7f6bb9e476`. Only `verify.py` reads `verification/benchmark.json`; ordinary training/inference do not read expected outputs. Fixtures contain public-file hashes and predictions, not private test labels. See [reproduction details](docs/REPRODUCTION.md).

Only the two final models in `models/` are distributed. Saved verification reports, test-run summaries, numerical-comparison results, logs and intermediate refits are excluded. Verification generates its predictions, reports and refitted models under the chosen `--output-dir`; use `artifacts/`, which Git ignores. The frozen `verification/benchmark.json` is retained as comparison input, not an execution log.

GitHub Actions is configured to run the dataset-free tests and file-checksum checks. Hosted CI has not been run as part of this local preparation.

## Repository map

| Path                          | Role                                                                   |
| ----------------------------- | ---------------------------------------------------------------------- |
| `run.py`                      | Ordinary CSV training, prediction and public-label evaluation CLI      |
| `flow_classifier/__init__.py` | Fixed task/class order and fold membership validation                  |
| `flow_classifier/features.py` | Shared features, protocol encoding and fitted preprocessing            |
| `flow_classifier/sizes.py`    | Integer size candidates, fingerprints, vocabulary and density          |
| `flow_classifier/kernel.py`   | Fixed similarity kernels, balanced ridge and reference support         |
| `flow_classifier/support.py`  | Class/protocol support, size interpolation, time model and context RFs |
| `flow_classifier/task1.py`    | Four content sources, OOF ranker and Task 1 prediction                 |
| `flow_classifier/task2.py`    | Three content sources, OOF fusion and Task 2 prediction                |
| `config/fitting.json`         | Unfitted recipes and explicit fold IDs                                 |
| `models/`                     | Two final 330-row models, usable directly                              |
| `verify.py`, `verification/`  | Optional benchmark reproduction                                        |
| `tests/`                      | Dataset-free public-interface and regression tests                     |
| `docs/`                       | Feature inventories and reproduction instructions                   |

Read the [technical report](TECHNICAL_REPORT.md) for the feature overview, architecture, training and evaluation; [feature inventory](docs/FEATURES.md) for all columns; [publication notes](PUBLICATION.md) for license/data attribution; and [citation metadata](CITATION.cff) for attribution to SPlab. `RELEASE_MANIFEST.json` records file checksums. Generated artifacts, datasets and environments are ignored by Git.
