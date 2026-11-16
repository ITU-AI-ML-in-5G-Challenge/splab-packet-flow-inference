# Optional reproduction and tests

Normal use consists of `run.py train`, `run.py predict` and, for public validation, `run.py evaluate`, as shown in [README.md](../README.md). They do not import the optional verifier or read expected predictions.

`verify.py` calls those same training/prediction commands. It adds automatic baseline comparisons and invariant checks rather than maintaining another learning implementation. Unit tests cover additional cases, including invalid inputs, unknown fingerprints and temporal-reference access. Neither check is required for ordinary training or inference.

## Files retained in the repository

| Path | Purpose |
|---|---|
| `tests/` | Dataset-free unit and standalone-interface tests |
| `verify.py` | Optional baseline reproduction using the ordinary CLI |
| `verification/benchmark.json` | Frozen input hashes, expected validation/test predictions, JSON hashes and selected-input hashes required for automatic comparisons |
| `config/fitting.json` | Selected unfitted estimator recipes and original inner-fold memberships |
| `models/` | Only the two final models fitted on all 330 labeled rows |
| `docs/FEATURES.md`, `docs/feature_columns.json` | Static feature definitions and inventories |

The benchmark's expected predictions are comparison inputs, not a disposable execution log. They contain no private test labels and are never used by normal training/inference. Saved verification reports, numerical-comparison results, test summaries, logs and intermediate model artifacts are not distributed.

## Dataset-free tests

```bash
python -m unittest discover -s tests -v
```

The 25 tests use synthetic examples and supplied final models. Competition CSVs are not required. They check input/fold contracts, shared preprocessing, RF routing and strictly past references, among other conditions.

## Reload the supplied models

```bash
python verify.py --data-dir "$DATA_DIR" \
  --output-dir artifacts/inference_check --inference-only
```

This checks the original CSV hashes and ten frozen selected-input checkpoints, then predicts the 81 test rows per task with the supplied 330-row models. It compares JSON bytes and labels, checks reverse/singleton invariance and audits temporal/RF references. It skips validation prediction because the supplied models were trained on those validation rows.

## Cold refit and compare

```bash
python verify.py --data-dir "$DATA_DIR" --output-dir artifacts/reproduction
```

This calls normal training twice per task: train 247 for validation 83, then train+validation 330 for test 81. It compares both prediction JSONs with frozen references, checks the selected inputs and invariants, and scores public validation. Supplied model files are not overwritten. Private test labels are unavailable; official scores are previously reported organizer results, not locally recomputed metrics.

Expected validation Macro F1/Accuracy are Task 1 .881581119/.939759036 and Task 2 .852153948/.915662651. Expected test JSON SHA-256 is `9c20120c13898bb1faae4974bc0d1b52eee78ab05189a82ca1a11e7f6bb9e476`. These are regression targets selected during historical development, not independent generalization estimates.

## Generated output stays outside release contents

Verification writes under the explicit `--output-dir`. The examples use the Git-ignored `artifacts/` directory:

```text
artifacts/reproduction/
├── validation/
│   ├── models/                 # refitted models and fitting summaries
│   └── predictions.json
├── test/
│   ├── models/                 # refitted models and fitting summaries
│   └── predictions.json
└── verification_report.json
```

With `--inference-only`, only the test prediction and verification report are generated. Console output can be redirected to a local log when needed. Generated predictions, fitting summaries, reports and refits are for the person executing the commands; they are not repository fixtures. Keep them under ignored output directories and leave the distributed `models/` unchanged.

## Release file integrity

The manifest covers source, configuration, supplied final models, documentation and frozen fixtures. GitHub Actions checks it before dataset-free tests; it does not download competition CSVs or execute cold reproduction.

After intentional edits, update hashes from the repository root. Add/remove manifest entries when release files are added/removed. Do not modify benchmark predictions to conceal a prediction regression.

```bash
python - <<'PYCODE'
import hashlib
import json
from pathlib import Path
path = Path("RELEASE_MANIFEST.json")
manifest = json.loads(path.read_text())
manifest["files"] = {
    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
    for name in manifest["files"]
}
path.write_text(json.dumps(manifest, indent=2) + "\n")
PYCODE
```

Checksums detect changed release bytes; they do not independently establish model quality. Detailed execution logs/results are generated locally rather than bundled in the repository.
