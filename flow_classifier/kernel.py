"""Fixed equal similarity kernels, balanced ridge and reference support."""

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from . import CLASSES
from . import features as feature_ops


def scores(model, query):
    q = np.asarray(query, float)
    centered = q - (q @ model["pi"])[:, None] - model["center"][None, :] + model["offset"]
    return centered @ model["coefficients"] + model["prior"]


def categorical_component(fit, query):
    def valid(value):
        return any(p not in ("NA", "nan", "None", "") for p in str(value).split("|")[1:])
    a = fit.iloc[:, 0].astype(str).to_numpy()
    b = query.iloc[:, 0].astype(str).to_numpy()
    va = np.asarray([valid(v) for v in a])
    vb = np.asarray([valid(v) for v in b])
    return ((b[:, None] == a[None, :]) & vb[:, None] & va[None, :]).astype(float)


def fit_numeric(fit, names):
    protected = [i for i, n in enumerate(names) if n.endswith(("_available", "_unique"))]
    scaled = [i for i in range(len(names)) if i not in protected]
    a = np.asarray(fit, float).copy()
    scaler = StandardScaler().fit(a[:, scaled])
    a[:, scaled] = np.clip(scaler.transform(a[:, scaled]), -5, 5)

    def family(name):
        if any((s in name for s in ("iat", "duration", "seconds", "burst"))):
            return "time"
        if any(
            (s in name for s in ("packet_count", "_available", "_unique", "log_candidates", "share", "ratio"))
        ):
            return "direction_count"
        return "size"

    groups = {
        g: [i for i, n in enumerate(names) if family(n) == g] for g in ("time", "direction_count", "size")
    }
    groups = {g: idx for g, idx in groups.items() if idx}
    return numeric_similarity(a, a, groups), dict(
        groups=groups, scaler=scaler, protected=protected, fit_numeric=a
    )


def numeric_similarity(query, fit, groups):
    distance = sum(
        cdist(query[:, ids], fit[:, ids], "sqeuclidean") / len(ids) for ids in groups.values()
    ) / len(groups)
    return np.exp(-distance)


def percentiles(values, reference):
    if not len(reference):
        return np.zeros_like(values, dtype=float)
    reference = np.sort(reference)
    return (np.searchsorted(reference, values, "left") + np.searchsorted(reference, values, "right")) / (2 * len(reference))


def support_weight(percentile):
    return 0.5 * np.clip(percentile / 0.25, 0, 1)


def top_two(matrix, eligible):
    a = np.where(eligible, matrix, -np.inf)
    enough = eligible.sum(1) >= 2
    result = np.zeros(len(a))
    result[enough] = np.sort(a[enough], axis=1)[:, -2:].mean(1)
    return (result, enough)


def fit_ridge(gram, labels, strength):
    k = np.asarray(gram, float)
    labels = np.asarray(labels)
    target = np.column_stack([labels == c for c in CLASSES]).astype(float)
    counts = target.sum(0)
    assert (counts > 0).all() and k.shape == (len(labels), len(labels))
    if strength <= 0:
        raise ValueError("Invalid learner")
    weights = (target / counts).sum(1) * len(labels) / len(CLASSES)
    pi = weights / weights.sum()
    root = np.sqrt(weights)
    center = k @ pi
    offset = float(pi @ center)
    centered = k - center[:, None] - center[None, :] + offset
    prior = pi @ target
    system = root[:, None] * centered * root[None, :] + strength * np.eye(len(labels))
    rhs = root[:, None] * (target - prior)
    beta = solve(system, rhs, assume_a="pos")
    np.testing.assert_allclose(system @ beta, rhs, atol=1e-09, rtol=1e-09)
    return dict(
        strength=float(strength),
        pi=pi,
        center=center,
        offset=offset,
        prior=prior,
        coefficients=root[:, None] * beta,
    )


def normalized(model, query):
    raw = scores(model, query)
    positive = np.maximum(raw, 0)
    total = positive.sum(1, keepdims=True)
    result = np.divide(positive, total, out=np.full_like(positive, 1 / len(CLASSES)), where=total > 0)
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-12)
    assert np.isfinite(result).all()
    return result


def global_support(gram, query, fit_protocol, query_protocol):
    """Same-protocol support of a query relative to leave-self-out training rows."""
    a, b = (np.asarray(fit_protocol), np.asarray(query_protocol))
    same = a[:, None] == a[None, :]
    np.fill_diagonal(same, False)
    reference, available = top_two(gram, same)
    values, query_ok = top_two(query, b[:, None] == a[None, :])
    result = np.zeros(len(b))
    for protocol in np.unique(a):
        mask = b == protocol
        result[mask] = support_weight(percentiles(values[mask], reference[(a == protocol) & available]))
    result[~query_ok] = 0
    return result


def components(numeric, categorical, protocol):
    proto = np.asarray(protocol)
    mask = proto[:, None] == proto[None, :]
    parts = [np.asarray(numeric)]
    for column in categorical:
        values = categorical[[column]]
        parts.append(categorical_component(values, values) * mask)
    return np.asarray(parts)


def fit_kernel(raw, structured, labels, preprocess, columns):
    pre = feature_ops.log_preprocessor(preprocess)
    matrix = pre.fit_transform(structured[list(columns)])
    names = list(pre.transformers_[0][2])
    nk, numeric = fit_numeric(matrix[:, :len(names)], names)
    proto = raw.transport.to_numpy()
    categories = list(pre.transformers_[2][2]) + ["struct_tcp_m", "struct_up_vocab"]
    cats = structured[categories].copy()
    mask = proto[:, None] == proto[None, :]
    parts = components(nk * mask, cats, proto)
    gram = np.tensordot(np.full(5, 0.2), parts, axes=1)
    return dict(preprocess=pre, columns=list(columns), numeric_names=names, numeric=numeric,
                fit_categories=cats, fit_transport=proto, fit_ids=raw.flow_id.tolist(),
                fit_labels=np.asarray(labels), fit_gram=gram, ridge=fit_ridge(gram, labels, 0.1))


def query_kernel(bundle, raw, frame, structured):
    a = bundle["preprocess"].transform(frame[bundle["columns"]])
    info = bundle["numeric"]
    width = len(bundle["numeric_names"])
    q = np.asarray(a[:, :width], float).copy()
    scaled = [i for i in range(width) if i not in info["protected"]]
    q[:, scaled] = np.clip(info["scaler"].transform(q[:, scaled]), -5, 5)
    nq = numeric_similarity(q, info["fit_numeric"], info["groups"])
    mask = raw.transport.to_numpy()[:, None] == bundle["fit_transport"][None, :]
    parts = [nq * mask]
    for column in bundle["fit_categories"]:
        cq = categorical_component(bundle["fit_categories"][[column]], structured[[column]])
        parts.append(cq * mask)
    return np.tensordot(np.full(5, 0.2), np.asarray(parts), axes=1)
