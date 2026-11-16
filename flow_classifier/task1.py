"""Task 1: content experts, OOF candidate ranker, and temporal adjustment."""

from catboost import CatBoostRanker
from catboost import Pool
import numpy as np
import copy
from sklearn.base import clone
from . import CLASSES, TASKS, fold_indices
from . import features as feature_ops
from . import sizes as size_ops
from . import kernel as kernel_ops
from . import support as support_ops

PARAMETERS = dict(
    iterations=100,
    depth=4,
    learning_rate=0.05,
    l2_leaf_reg=5.0,
    loss_function="PairLogit",
    boosting_type="Plain",
    bootstrap_type="No",
    thread_count=1,
    random_seed=1741,
    allow_writing_files=False,
    verbose=False,
)
MODELS = ("compact", "full_cat", "hgb", "ridge")


def margin(p):
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def fit_candidate_ranker(data, labels):
    labels = np.asarray(labels)
    n = len(labels)
    if data["x"].ndim != 3 or data["x"].shape[:2] != (n, len(CLASSES)):
        raise ValueError("Ranker input must contain six ordered candidates per labeled flow")
    x = data["x"].reshape(n * 6, -1)
    truth = np.array([CLASSES.index(str(c)) for c in labels])
    target = (np.arange(6)[None, :] == truth[:, None]).astype(float).ravel()
    pairs = [(i * 6 + int(truth[i]), i * 6 + j) for i in range(n) for j in range(6) if j != truth[i]]
    pool = Pool(
        x,
        label=target,
        group_id=np.repeat(np.arange(n), 6),
        pairs=pairs,
        pairs_weight=np.full(len(pairs), 1 / 5),
    )
    return CatBoostRanker(**PARAMETERS).fit(pool)


def join_candidate_rows(blocks):
    return {name: np.concatenate([b[name] for b in blocks], axis=0) for name in ("x", "default", "votes")}


def features(scores, cs, geometry, physical):
    names = MODELS
    n = len(scores["compact"])
    row = np.arange(n)
    mixed = scores["compact"].copy()
    use = scores["support"] >= 0.5
    mixed[use] = 0.5 * mixed[use] + 0.5 * scores["ridge"][use]
    stack = np.stack([scores[name] for name in names] + [mixed], axis=1)
    default = mixed.argmax(1)
    confidence = stack.max(2)
    gaps = np.column_stack([margin(stack[:, h]) for h in range(stack.shape[1])])
    ordered = np.sort(stack, axis=2)
    entropy = -(ordered * np.log(np.maximum(ordered, 1e-12))).sum(2)
    common = [confidence, gaps, entropy, scores["support"]]
    common.extend([scores["missing_direction"], physical])
    common = np.column_stack(common)
    baseline = stack[row, :, default]
    baseline_geometry = geometry[row, default]
    blocks, votes = [], []
    for candidate in range(len(CLASSES)):
        own = stack[:, :, candidate]
        other = np.delete(stack, candidate, axis=2).max(2)
        agree = (own == confidence) & (gaps > 0)
        fields = [common, own, own - other, agree, baseline, own - baseline, candidate == default]
        proposed_geometry = geometry[:, candidate]
        fields.extend(
            [
                cs[:, candidate],
                proposed_geometry,
                baseline_geometry,
                proposed_geometry - baseline_geometry,
            ]
        )
        blocks.append(np.column_stack(fields))
        votes.append(agree[:, : len(names)].sum(1))
    x = np.stack(blocks, axis=1)
    assert np.isfinite(x).all()
    return dict(x=x, default=default, votes=np.column_stack(votes), mixed=mixed)


def select(data, raw_quality, config):
    n = len(data["default"])
    quality = np.asarray(raw_quality).reshape(n, len(CLASSES))
    eligible = data["eligible"]
    row = np.arange(n)
    value = np.where(eligible, quality, -np.inf)
    candidate = value.argmax(1)
    out = data["default"].copy()
    allowed = value[row, candidate] > quality[row, out] + config["margin"]
    out[allowed] = candidate[allowed]
    return np.array(CLASSES)[out]


def class_geometry(base, raw, query):
    k = base["kernel"]
    q = query
    fitproto = k["fit_transport"]
    labels = k["fit_labels"]
    qp = raw.transport.to_numpy()
    n = len(raw)
    near = np.zeros((n, 6))
    percentile = np.zeros((n, 6))
    counts = np.zeros((n, 6))
    average = np.zeros((n, 6))
    available = np.zeros((n, 6))
    fitmask = (fitproto[:, None] == fitproto[None, :]) & (labels[:, None] == labels[None, :])
    np.fill_diagonal(fitmask, False)
    ref, refok = kernel_ops.top_two(k["fit_gram"], fitmask)
    for j, c in enumerate(CLASSES):
        querymask = (qp[:, None] == fitproto[None, :]) & (labels[None, :] == c)
        value, valid = kernel_ops.top_two(q, querymask)
        near[:, j] = value
        available[:, j] = valid
        counts[:, j] = querymask.sum(1)
        average[:, j] = np.divide(
            (q * querymask).sum(1), counts[:, j], out=np.zeros(n), where=counts[:, j] > 0
        )
        for protocol in np.unique(fitproto):
            ix = qp == protocol
            reference = ref[(fitproto == protocol) & (labels == c) & refok]
            percentile[ix, j] = kernel_ops.percentiles(value[ix], reference)
    prior = np.divide(
        counts,
        counts.sum(1, keepdims=True),
        out=np.zeros_like(counts),
        where=counts.sum(1, keepdims=True) > 0,
    )
    contrast = np.empty_like(near)
    for j in range(6):
        contrast[:, j] = near[:, j] - np.delete(near, j, axis=1).max(1)
    return np.stack([near, percentile, prior, average, available, contrast], axis=2)


def evidence(bundle, raw, prefix, frame, query, body, structured):
    base = bundle["baseline"]
    models = base["models"]
    kernel = bundle["kernel"]
    return dict(
        compact=feature_ops.component_probabilities(models["compact"], frame),
        full_cat=feature_ops.component_probabilities(models["full_cat"], structured),
        hgb=models["hgb"].predict_proba(body[base["body_columns"]]),
        ridge=kernel_ops.normalized(kernel["ridge"], query),
        support=kernel_ops.global_support(kernel["fit_gram"], query, kernel["fit_transport"], raw.transport.to_numpy()),
        missing_direction=(prefix.packet_count_up.to_numpy() == 0) | (prefix.packet_count_down.to_numpy() == 0),
    )


def context(frame, body):
    """Exactly the 190 physical/missingness columns used by the ranker.

    The previous code built seven candidate matrices and discarded all but
    these columns. Build them directly, retaining order and float operations.
    """
    columns = [
        "duration",
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
        "shape_pl_cv_up",
        "shape_pl_cv_down",
        "shape_packet_up_share",
        "shape_byte_up_share",
        "shape_iat_cv",
        "shape_iat_max_duration",
        "active_iat_mean",
        "active_seconds_per_kib",
        "trimmed_iat_cv",
        "p_duration",
        "p_packet_count_up",
        "p_packet_count_down",
        "p_bytes_up",
        "p_bytes_down",
        "p_mean_pkt_size_up",
        "p_mean_pkt_size_down",
        "p_std_pkt_size_up",
        "p_std_pkt_size_down",
        "p_mean_iat",
        "p_std_iat",
        "p_min_iat",
        "p_max_iat",
        "p_burst_count",
        "p_up_down_packet_ratio",
        "p_up_down_byte_ratio",
        "p_up_max_lower",
        "p_up_max_upper",
        "p_up_min_lower",
        "p_up_min_upper",
        "p_up_exact_extrema_available",
        "p_up_exact_min",
        "p_up_exact_max",
        "p_down_max_lower",
        "p_down_max_upper",
        "p_down_min_lower",
        "p_down_min_upper",
        "p_down_exact_extrema_available",
        "p_down_exact_min",
        "p_down_exact_max",
        "r_byte_up_share",
        "r_up_mean_size",
        "r_down_mean_size",
        "r_after5_seconds",
        "p_byte_share",
    ]
    values = np.column_stack([frame[columns].to_numpy(float), body.to_numpy(float)])
    valid = np.isfinite(values)
    values = np.where(valid, values, 0)
    values = np.sign(values) * np.log1p(np.abs(values))
    result = np.column_stack([values, valid])
    assert result.shape == (len(frame), 2 * (len(columns) + body.shape[1]))
    return result


def data_for(base, raw, prefix):
    frame = feature_ops.make_selected_frame(TASKS[0], raw, prefix)
    baseline = base["baseline"]
    structured = feature_ops.structural_query(frame, size_ops.structural_catalog(prefix), baseline["vocabulary"])
    query = kernel_ops.query_kernel(base["kernel"], raw, frame, structured)
    body = feature_ops.compact_features(TASKS[0], raw, prefix)
    scores = evidence(base, raw, prefix, frame, query, body, structured)
    geometry = class_geometry(base, raw, query)
    cs = kernel_ops.support_weight(geometry[:, :, 1])
    cs[geometry[:, :, 4] == 0] = 0
    return features(scores, cs, geometry, context(frame, body))


def predict(bundle, raw, prefix):
    parent = bundle["parent"]
    q = data_for(parent["base"], raw, prefix)
    q["eligible"] = q["votes"] >= parent["config"]["minimum_votes"]
    q["eligible"][np.arange(len(raw)), q["default"]] = True
    quality = parent["selector"].predict(q["x"].reshape(len(raw) * len(CLASSES), -1))
    old = select(q, quality, parent["config"])
    result = support_ops.adjust_time(
        quality.reshape(len(raw), len(CLASSES)), old, bundle["time"], raw, 0.5, q["eligible"]
    )
    return support_ops.limit_protocol_changes(
        bundle["protocol_guard"], raw, np.asarray(CLASSES)[q["default"]], result
    )


def fresh_component(spec, frame, y):
    pre = clone(spec["preprocess"])
    a = pre.fit_transform(frame[spec["columns"]])
    model = (
        type(spec["model"])(**copy.deepcopy(spec["model"].get_params()))
        if spec["kind"] in ("cat", "full_cat")
        else clone(spec["model"])
    )
    if spec["kind"] in ("cat", "full_cat"):
        x, cats = feature_ops.native(pre, frame, a)
        assert cats == spec["cats"]
    else:
        x = a
    model.fit(x, y)
    np.testing.assert_array_equal(model.classes_, CLASSES)
    return dict(
        spec,
        preprocess=pre,
        model=model,
    )


def fit_primary(recipe, raw, prefix, y):
    frame = feature_ops.make_selected_frame(TASKS[0], raw, prefix)
    structured, vocab = feature_ops.structural(frame, size_ops.structural_catalog(prefix))
    body = feature_ops.compact_features(TASKS[0], raw, prefix)
    models = dict(
        compact=fresh_component(recipe["models"]["compact"], frame, y),
        full_cat=fresh_component(recipe["models"]["full_cat"], structured, y),
        hgb=clone(recipe["models"]["hgb"]).fit(body[recipe["body_columns"]], y),
    )
    central = recipe["models"]["compact"]
    kernel = kernel_ops.fit_kernel(raw, structured, y, central["preprocess"], central["columns"])
    return dict(baseline=dict(models=models, body_columns=recipe["body_columns"], vocabulary=vocab), kernel=kernel)


def fit(recipe, raw, prefix, y, scope):
    parts = []
    labels = []
    seen = []
    for i, e in enumerate(recipe["memberships"][scope]):
        fi, he = fold_indices(raw, e)
        primary = fit_primary(recipe, raw.iloc[fi], prefix.iloc[fi], y[fi])
        part = data_for(primary, raw.iloc[he], prefix.iloc[he])
        parts.append(part)
        labels.extend(y[he])
        seen += e["held_ids"]
        print("cold Task1", scope, "inner", i + 1, "/3", flush=True)
    assert len(seen) == len(set(seen)) == len(raw) and set(seen) == set(raw.flow_id)
    training = join_candidate_rows(parts)
    selector = fit_candidate_ranker(training, labels)
    primary = fit_primary(recipe, raw, prefix, y)
    parent = dict(base=primary, selector=selector, config=recipe["config"], fit_ids=raw.flow_id.tolist())
    bundle = dict(
        parent=parent,
        time=support_ops.fit_time(raw, y),
        fit_ids=raw.flow_id.tolist(),
        fit_labels=y,
    )
    bundle["protocol_guard"] = support_ops.fit_protocol_guard(raw, y, recipe["minimum_protocol_rows"])
    return bundle
