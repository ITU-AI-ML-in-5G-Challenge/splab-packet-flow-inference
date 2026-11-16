# Packet-flow application classification

**SPlab — ITU Application Inference from Packet Flows Challenge**

## 1. Task and data

The solution classifies flows into Facebook, Google, Instagram, Mozilla, X or YouTube using the released aggregate CSVs. Task 1 combines full-flow statistics with the corresponding first-five-packet statistics. Task 2 uses only the first-five-packet data. Payloads, addresses, domain names and metadata class-count constraints are not used. Flow IDs align records and check overlap; they are not predictor features.

Each task has 247 training flows, 83 labeled validation flows and 81 official test flows. Validation evaluation fits the 247 training flows; the distributed final models fit all 330 public labeled flows.

## 2. Features and preprocessing

Task 1's main ExtraTrees (ET) uses 61 columns before categorical expansion:

| Feature group | Columns |
|---|---:|
| Full-flow statistics and shape | 24 |
| Raw first-five-packet statistics | 16 |
| Packet-size bounds and exact extrema where available | 14 |
| Size fingerprint categories | 2 |
| Residual statistics and prefix byte share | 5 |

These describe duration, directional counts/bytes/size moments, IAT, bursts, directional balance and the relationship between the prefix and remaining flow. Packet-size bounds and compatible integer size candidates are derived from counts, byte sums, means and population standard deviations. Ambiguous candidates do not establish packet order or protocol-message identity. Task 1's CatBoost adds two structural categories (63 inputs); HGB selects 17 body/statistical inputs.

Task 2 uses 33 columns: 16 raw prefix statistics, transport, 14 size-bound/extrema features and two fingerprints. Integer size candidates also support its latent-size model.

Missing numerical inputs use fitting-data medians where required. Task 1 applies signed log1p; Task 2 applies a fitted uniform quantile transform. TCP/UDP is encoded as 0/1. ET uses one-hot categories and CatBoost uses native categories. Exact feature definitions are in [docs/FEATURES.md](docs/FEATURES.md).

## 3. Model structure and training

**Task 1.** ET, CatBoost, histogram gradient boosting (HGB) and kernel ridge provide complementary evidence. A support check determines whether the ET default is combined with ridge scores. A learned ranker compares all six candidate classes using model scores, agreement and statistical context. Changes require supporting evidence, followed by temporal correction and class/protocol support checks.

**Task 2.** ET, CatBoost and a latent packet-size density model are combined using learned log-probability weights and class offsets. Temporal correction and class/protocol support checks are followed by downstream-size interpolation and past-context RFs within supported signatures. The final method has no application-pair or query-ID-specific override.

Both integrations are trained using three-fold inner out-of-fold (OOF) predictions. Each held-out row's content evidence is produced without fitting on that row, including preprocessing and vocabulary. After integration training, the content models are refitted on the complete supplied scope. Selected settings and fold memberships are stored in [config/fitting.json](config/fitting.json).

Both tasks use same-recording `start_rel` context and historical **training labels**. Query-time temporal messages and Task 2 RF references use strictly earlier starts, excluding equal/later starts. Global models and temporal parameters are nevertheless fitted offline on the full training scope. This assumes a shared recording/time origin and historical labels; queries do not exchange predictions or labels.

## 4. Class imbalance

ET, HGB and context RFs use balanced class weights; CatBoost uses automatic balanced weights. Ridge uses inverse class-frequency weights. The ranker uses equal pair weights, while fusion and temporal likelihood are not class-balanced losses. No synthetic minority examples or metadata count matching are used. Macro F1 evaluates the six classes equally.

## 5. Evaluation

| Evaluation | Task 1 Macro F1 | Task 1 Accuracy | Task 2 Macro F1 | Task 2 Accuracy |
|---|---:|---:|---:|---:|
| Fit 247 → validation 83 | 0.881581 | 0.939759 | 0.852154 | 0.915663 |
| Fit 330 → official test 81 | **0.971178** | **0.962963** | **0.951341** | **0.938272** |

The reported official total is **96.125910 / 100**. Official scores were supplied by the organizers; private test labels are unavailable locally. Validation data and repeated official feedback informed selection, so these are not untouched model-selection holdout results. Inner OOF constructs integration inputs rather than providing an independent full-method evaluation.

Reproduction preserves the submitted prediction JSONs. Optional tests and verification call or inspect the ordinary implementation; instructions are in [README.md](README.md) and [docs/REPRODUCTION.md](docs/REPRODUCTION.md). The repository includes the two final models and frozen comparison inputs, while generated reports, logs and intermediate refits remain outside the distributed files.

## 6. Contribution

The method combines moment-constrained packet fingerprints, OOF-trained model integration and protocol-aware recording context. It uses uncertain size information without asserting a unique packet sequence and limits auxiliary changes to fitted support. The underlying classifiers, ridge and ranking methods are established techniques; the contribution is their task-specific representation and integration. Structural hypotheses and support thresholds are empirical design choices, not universal protocol rules.
