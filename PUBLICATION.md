# License, data and attribution

## Solution license

Copyright (c) 2026 SPlab. The original solution code, documentation and supplied trained model artifacts are released under the [MIT License](LICENSE). Attribute the solution to SPlab; citation metadata is provided in [CITATION.cff](CITATION.cff).

## Challenge data

The [ITU challenge](https://competition.aiforgood.itu.int/web/challenges/challenge-page/495/overview) and [official dataset/starter repository](https://github.com/FG-AI4H/application-inference-packet-flows-public) are the data and evaluation sources. Competition CSVs are not included. Obtain them separately and follow the provider's terms. This solution license does not grant rights in, or relicense, the underlying challenge data.

The trained artifacts contain fitting-derived reference vectors, class labels, flow IDs and recording-time information used by support and temporal components. They are not reference-free estimators. This is stated explicitly so downstream users know what the model release contains. Verification fixtures also contain public-data hashes and the solution's predictions; they contain no private test labels.

## Dependencies

NumPy, SciPy, pandas, scikit-learn, CatBoost, joblib, threadpoolctl and transitive dependencies retain their own licenses and notices. They are installed from their distributions and are not vendored or relicensed here. Requirements files specify the tested dependencies; consult the corresponding distributions for their license texts.

## Repository scope

Publish the contents of this solution directory as the repository root. Do not upload its parent research workspace, historical output directories, datasets, local environments or backup archives. Normal execution does not depend on any of them. Model files are small enough to be included directly; no model download service is used.

The local preparation does not create a GitHub repository, upload files or run hosted CI. Those are separate publication actions.
