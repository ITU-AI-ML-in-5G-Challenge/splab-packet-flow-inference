"""Row-local features and fitted input preparation shared by both tasks."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from . import CLASSES, TASKS
from . import sizes as size_ops


def reconstruct(raw):
    if not (raw.packet_count_up + raw.packet_count_down).eq(5).all():
        raise ValueError("Exactly four interarrival gaps required")
    duration, mean, sd, minimum, maximum = [
        raw[c].to_numpy(float) for c in ("duration", "mean_iat", "std_iat", "min_iat", "max_iat")
    ]
    eps = 5e-10
    if (abs(4 * mean - duration) > 5 * eps).any():
        raise ValueError("Duration and gap mean disagree")
    s = duration - minimum - maximum
    q = 4 * (mean * mean + sd * sd) - minimum * minimum - maximum * maximum
    delta = 2 * q - s * s
    e_s = 3 * eps
    e_q = 8 * eps * (abs(mean) + abs(sd)) + 2 * eps * (abs(minimum) + abs(maximum)) + 10 * eps * eps
    e_d = 2 * e_q + 2 * abs(s) * e_s + e_s * e_s + 64 * np.finfo(float).eps * np.maximum(abs(q) + s * s, 1)
    if (delta < -e_d).any():
        raise ValueError("No compatible middle-gap difference")
    root = np.sqrt(np.maximum(delta, 0))
    lower, upper = ((s - root) / 2, (s + root) / 2)
    width = (e_s + np.sqrt(np.maximum(delta + e_d, 0)) - np.sqrt(np.maximum(delta - e_d, 0))) / 2
    if ((lower < minimum - width - 1e-09) | (upper > maximum + width + 1e-09)).any():
        raise ValueError("Middle gaps incompatible with min/max")
    clipped_lower = np.clip(lower, minimum, maximum)
    clipped_upper = np.clip(upper, minimum, maximum)
    return pd.DataFrame(
        dict(
            iat_middle_lower_estimate=clipped_lower,
            iat_middle_upper_estimate=clipped_upper,
            iat_middle_error_bound=width,
        ),
        index=raw.index,
    )


TRANSPORT_CODES = {"tcp": 0, "udp": 1}


def safe_ratio(numerator, denominator):
    denominator = np.asarray(denominator, dtype=float)
    numerator = np.asarray(numerator, dtype=float)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


FULL = (
    "f_duration",
    "f_bursts_per_second",
    "f_packets_per_burst",
    "f_down_mean",
    "f_down_std",
    "f_up_byte_share",
)
REST = (
    "r_duration",
    "r_burst_difference_per_second",
    "r_packets",
    "r_down_mean",
    "r_down_std",
    "r_up_byte_share",
)


def finite_ratio(a, b):
    a, b = (np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


def signed_log(x):
    return signed_log1p(x)


HIGH = "prefix_iat_middle_upper"
LOW = "prefix_iat_middle_lower"


def keys(raw):
    """The selected upstream-byte fingerprint; no alternative profiles."""
    columns = ["transport", "packet_count_up", "packet_count_down", "bytes_up"]
    return list(raw[columns].itertuples(index=False, name=None))


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


EPS_CSV = 5e-10
EXTREMA_COLUMNS = tuple(
    (
        f"p_{direction}_{name}"
        for direction in ("up", "down")
        for name in (
            "max_lower",
            "max_upper",
            "min_lower",
            "min_upper",
            "exact_extrema_available",
            "exact_min",
            "exact_max",
        )
    )
)
PREFIX_COLUMNS = tuple(
    (
        "p_" + c
        for c in (
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
            "up_down_packet_ratio",
            "up_down_byte_ratio",
        )
    )
)
def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b != 0)


def body_statistics(full, prefix):
    if not full.flow_id.equals(prefix.flow_id):
        raise ValueError("Full/prefix IDs must be aligned")
    n = full.packet_count_up + full.packet_count_down
    tb = full.bytes_up + full.bytes_down
    d = full.duration - prefix.duration
    rn = n - prefix.packet_count_up - prefix.packet_count_down
    nb = full.burst_count - prefix.burst_count
    up = full.bytes_up - prefix.bytes_up
    down = full.bytes_down - prefix.bytes_down
    nd = full.packet_count_down - prefix.packet_count_down
    if any((np.any(np.asarray(x) < -1e-08) for x in (d, rn, nb, up, down, nd))):
        raise ValueError("Prefix must be contained in full flow")
    mean = finite_ratio(down, nd)
    q = full.packet_count_down * (full.std_pkt_size_down**2 + full.mean_pkt_size_down**2)
    q -= prefix.packet_count_down * (prefix.std_pkt_size_down**2 + prefix.mean_pkt_size_down**2)
    variance = finite_ratio(q, nd) - mean**2
    if np.any(variance[np.isfinite(variance)] < -1e-05):
        raise ValueError("Negative residual variance")
    std = np.sqrt(np.maximum(variance, 0))
    std[np.asarray(nd) == 1] = 0
    out = pd.DataFrame(
        dict(
            f_duration=full.duration,
            f_bursts_per_second=finite_ratio(full.burst_count, full.duration),
            f_packets_per_burst=finite_ratio(n, full.burst_count),
            f_down_mean=np.where(full.packet_count_down > 0, full.mean_pkt_size_down, np.nan),
            f_down_std=np.where(full.packet_count_down > 0, full.std_pkt_size_down, np.nan),
            f_up_byte_share=finite_ratio(full.bytes_up, tb),
            r_duration=d,
            r_burst_difference_per_second=finite_ratio(nb, d),
            r_packets=rn,
            r_down_mean=mean,
            r_down_std=std,
            r_up_byte_share=finite_ratio(up, up + down),
        ),
        index=full.index,
    )
    out.loc[rn.eq(0), list(REST)] = np.nan
    if np.isinf(out.to_numpy()).any():
        raise ValueError("Infinite features")
    return out[list(FULL + REST)]


def middle_gap_statistics(raw):
    a = reconstruct(raw)
    duration = raw.duration.to_numpy(float)
    total = duration - raw.min_iat.to_numpy(float) - raw.max_iat.to_numpy(float)
    lo = np.maximum(raw.min_iat.to_numpy(float), total - raw.max_iat.to_numpy(float))
    hi = np.minimum(raw.max_iat.to_numpy(float), total / 2)
    width = a.iat_middle_error_bound.to_numpy(float)
    if (lo > hi + width + 1e-09).any():
        raise ValueError("Incompatible central gap sum")
    lo = np.minimum(lo, hi)
    lower = np.clip(a.iat_middle_lower_estimate.to_numpy(float), lo, hi)
    upper = total - lower
    return pd.DataFrame(
        {
            LOW: lower,
            HIGH: upper,
        },
        index=raw.index,
    )


def residual_moments(nf, mf, sf, np_, mp, sp, residual_sum):
    """Subtract additive moments, checking CSV-rounding error, not hiding errors."""
    nf, mf, sf, np_, mp, sp, residual_sum = [
        np.asarray(x, float) for x in (nf, mf, sf, np_, mp, sp, residual_sum)
    ]
    nr = nf - np_
    if (nr < 0).any():
        raise ValueError("Prefix has more observations than full flow")
    mean = divide(residual_sum, nr)
    qf, qp = (nf * (sf**2 + mf**2), np_ * (sp**2 + mp**2))
    m2 = qf - qp - nr * mean**2
    tol = (
        nf * (2 * (abs(sf) + abs(mf)) * EPS_CSV + 2 * EPS_CSV**2)
        + np_ * (2 * (abs(sp) + abs(mp)) * EPS_CSV + 2 * EPS_CSV**2)
        + 4 * abs(np.nan_to_num(mean)) * EPS_CSV
        + 32 * np.finfo(float).eps * (abs(qf) + abs(qp))
    )
    bad = (nr > 0) & (m2 < -tol)
    if bad.any():
        raise ValueError(f"Incompatible residual population moments: {np.where(bad)[0].tolist()}")
    m2 = np.where(abs(m2) <= tol, 0.0, m2)
    variance = divide(np.maximum(m2, 0), nr)
    variance = np.where(nr == 1, 0.0, variance)
    return (mean, np.sqrt(variance))


def compact_features(task, raw, prefix=None):
    if task != TASKS[0]:
        raise ValueError("Compact body features are used only by Task 1")
    p = prefix
    if p is None:
        raise ValueError("Task1 requires corresponding prefix")
    if not raw.flow_id.equals(p.flow_id):
        raise ValueError("Raw/prefix IDs differ")
    out = pd.DataFrame(index=raw.index)
    out["transport"] = p.transport.map({"tcp": 0.0, "udp": 1.0})
    if out.transport.isna().any():
        raise ValueError("Unknown transport")
    for dest, src in (
        ("p_duration", "duration"),
        ("p_nup", "packet_count_up"),
        ("p_bup", "bytes_up"),
        ("p_bdown", "bytes_down"),
        ("p_iat_mean", "mean_iat"),
        ("p_iat_std", "std_iat"),
        ("p_iat_max", "max_iat"),
        ("p_bursts", "burst_count"),
    ):
        out[dest] = p[src]
    for dr in ("up", "down"):
        available = p[f"packet_count_{dr}"] > 0
        out[f"p_mean_{dr}"] = np.where(available, p[f"mean_pkt_size_{dr}"], np.nan)
        out[f"p_std_{dr}"] = np.where(available, p[f"std_pkt_size_{dr}"], np.nan)
        out[f"p_{dr}_available"] = available.astype(float)
        out[f"p_cv_{dr}"] = finite_ratio(out[f"p_std_{dr}"], out[f"p_mean_{dr}"])
    out["p_up_byte_share"] = finite_ratio(p.bytes_up, p.bytes_up + p.bytes_down)
    out["p_iat_cv"] = finite_ratio(p.std_iat, p.mean_iat)
    out["p_max_gap_fraction"] = finite_ratio(p.max_iat, p.duration)
    out["p_mean_contrast"] = finite_ratio(out.p_mean_up - out.p_mean_down, out.p_mean_up + out.p_mean_down)
    body = body_statistics(raw, p)
    for col in body:
        out[col] = body[col]
    out["f_max_iat"] = raw.max_iat
    out["f_iat_cv"] = finite_ratio(raw.std_iat, raw.mean_iat)
    out["f_max_gap_fraction"] = finite_ratio(raw.max_iat, raw.duration)
    out["r_down_cv"] = finite_ratio(body.r_down_std, body.r_down_mean)
    if np.isinf(out.to_numpy()).any():
        raise ValueError("Infinite compact features")
    assert not set(out) & {"flow_id", "start_rel", "split", "app", "server_port"}
    return out


def coordinates(raw):
    middle = middle_gap_statistics(raw)
    g = np.column_stack([raw.min_iat, middle[LOW], middle[HIGH], raw.max_iat]).astype(float)
    assert np.isfinite(g).all() and g.min() >= -1e-09
    g = np.maximum(g, 0)
    g = np.sort(g, axis=1)
    np.testing.assert_allclose(g.sum(1), raw.duration.to_numpy(float), atol=2e-09, rtol=0)
    return np.log1p(g / 0.001)


def build_feature_frame(raw):
    """Validate raw moments and compute only the six selected shape inputs."""
    numeric = ["duration", "mean_iat", "std_iat", "min_iat", "max_iat"]
    for direction in ("up", "down"):
        numeric.extend(f"{name}_{direction}" for name in ("packet_count", "bytes", "mean_pkt_size", "std_pkt_size"))
    values = raw[numeric].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Statistics must be finite and nonnegative")
    for direction in ("up", "down"):
        count = raw[f"packet_count_{direction}"].to_numpy(float)
        total = raw[f"bytes_{direction}"].to_numpy(float)
        mean = raw[f"mean_pkt_size_{direction}"].to_numpy(float)
        sd = raw[f"std_pkt_size_{direction}"].to_numpy(float)
        if (count != np.floor(count)).any() or (total != np.floor(total)).any():
            raise ValueError("Packet counts and byte totals must be integers")
        if ((count == 0) & ((total != 0) | (mean != 0) | (sd != 0))).any():
            raise ValueError("Absent directions must have zero statistics")
        tolerance = np.maximum(1e-12 * np.maximum(abs(total), abs(mean * count)), count * EPS_CSV + 1e-8)
        if (abs(total - count * mean) > tolerance).any():
            raise ValueError("Bytes disagree with mean times packet count")
    count = raw.packet_count_up + raw.packet_count_down
    total = raw.bytes_up + raw.bytes_down
    if (count <= 1).any():
        raise ValueError("At least two packets are required for IAT statistics")
    result = raw.copy()
    derived = {
        "shape_pl_cv_up": safe_ratio(raw.std_pkt_size_up, raw.mean_pkt_size_up),
        "shape_pl_cv_down": safe_ratio(raw.std_pkt_size_down, raw.mean_pkt_size_down),
        "shape_iat_cv": safe_ratio(raw.std_iat, raw.mean_iat),
        "shape_iat_max_duration": safe_ratio(raw.max_iat, raw.duration),
        "shape_packet_up_share": safe_ratio(raw.packet_count_up, count),
        "shape_byte_up_share": safe_ratio(raw.bytes_up, total),
    }
    for name, values in derived.items():
        if name in result and not np.allclose(result[name], values, equal_nan=True):
            raise ValueError(f"Existing derived feature conflicts with calculation: {name}")
        result[name] = values
    return result


def feature_frame(raw):
    """Generate selected shape and active-time features without experiment groups."""
    frame = build_feature_frame(raw)
    active = np.maximum(frame.duration - frame.max_iat, 0)
    additions = {
        "active_iat_mean": safe_ratio(active, frame.packet_count_up + frame.packet_count_down - 2),
        "active_seconds_per_kib": safe_ratio(active, (frame.bytes_up + frame.bytes_down) / 1024),
    }
    n = frame.packet_count_up + frame.packet_count_down - 1
    mean = frame.mean_iat
    maximum = frame.max_iat
    trimmed_mean = safe_ratio(n * mean - maximum, n - 1)
    numerator = n * frame.std_iat**2 - (maximum - mean) * (maximum - trimmed_mean)
    variance = safe_ratio(numerator, n - 1)
    scale = safe_ratio(n * frame.std_iat**2 + abs((maximum - mean) * (maximum - trimmed_mean)), n - 1)
    tolerance = 1e-10 + 1e-06 * scale
    valid = (n >= 2) & (trimmed_mean >= -1e-09) & (variance >= -tolerance)
    std = np.where(valid, np.sqrt(np.maximum(variance, 0)), np.nan)
    additions["trimmed_iat_cv"] = safe_ratio(std, np.maximum(trimmed_mean, 0))
    for name, values in additions.items():
        if name in frame and (not np.allclose(frame[name], values, equal_nan=True)):
            raise ValueError("Existing feature conflicts with formula: " + name)
        frame[name] = values
    return frame


def prefix_size_bounds(raw):
    """Shared 14 size bounds/exact extrema for either task's same prefix."""
    frame = pd.DataFrame(index=raw.index)
    for direction in ("up", "down"):
        n = raw[f"packet_count_{direction}"].to_numpy(float)
        mean = raw[f"mean_pkt_size_{direction}"].to_numpy(float)
        std = raw[f"std_pkt_size_{direction}"].to_numpy(float)
        if (std < 0).any() or not np.isfinite(mean + std).all():
            raise ValueError("Invalid prefix packet moments")
        root = np.sqrt(np.maximum(n - 1, 0))
        spread = np.divide(std, root, out=np.zeros_like(std), where=root != 0)
        values = dict(
            max_lower=mean + spread,
            max_upper=mean + std * root,
            min_lower=mean - std * root,
            min_upper=mean - spread,
        )
        for name, v in values.items():
            frame[f"p_{direction}_{name}"] = np.where(n > 0, np.maximum(v, 0), np.nan)
        available = (n == 1) | (n == 2)
        frame[f"p_{direction}_exact_extrema_available"] = available.astype(float)
        frame[f"p_{direction}_exact_min"] = np.where(available, mean - std, np.nan)
        frame[f"p_{direction}_exact_max"] = np.where(available, mean + std, np.nan)
    return frame[list(EXTREMA_COLUMNS)]


def paired_frame(full, prefix):
    """Build the full-flow design and five selected residual/prefix-share inputs."""
    for raw in (full, prefix):
        if raw.flow_id.duplicated().any() or raw.flow_id.isna().any():
            raise ValueError("Flow IDs must be unique and nonmissing")
    if set(full.flow_id) != set(prefix.flow_id):
        raise ValueError("Full/prefix ID sets differ")
    p = prefix.set_index("flow_id").loc[full.flow_id].reset_index()
    for column in ("transport", "server_port", "start_rel", "split"):
        if not np.array_equal(full[column].to_numpy(), p[column].to_numpy()):
            raise ValueError(f"Paired context differs: {column}")
    f = full.reset_index(drop=True)
    frame = feature_frame(f)
    for name in PREFIX_COLUMNS:
        frame[name] = p[name[2:]].to_numpy()
    full_n = (f.packet_count_up + f.packet_count_down).to_numpy(float)
    pre_n = (p.packet_count_up + p.packet_count_down).to_numpy(float)
    if not np.all(pre_n == 5) or not np.all(full_n >= pre_n):
        raise ValueError("Expected first-five-packet subset")
    for column, values in prefix_size_bounds(p).items():
        frame[column] = values.to_numpy()
    remaining = {}
    for direction in ("up", "down"):
        nf = f[f"packet_count_{direction}"].to_numpy(float)
        n = p[f"packet_count_{direction}"].to_numpy(float)
        br = (f[f"bytes_{direction}"] - p[f"bytes_{direction}"]).to_numpy(float)
        if (br < 0).any() or ((nf == n) & (br != 0)).any():
            raise ValueError("Invalid remaining bytes")
        mean, _ = residual_moments(
            nf, f[f"mean_pkt_size_{direction}"], f[f"std_pkt_size_{direction}"],
            n, p[f"mean_pkt_size_{direction}"], p[f"std_pkt_size_{direction}"], br
        )
        frame[f"r_{direction}_mean_size"] = mean
        remaining[direction] = br
    full_b = (f.bytes_up + f.bytes_down).to_numpy(float)
    pre_b = (p.bytes_up + p.bytes_down).to_numpy(float)
    dt = (f.duration - p.duration).to_numpy(float)
    if (dt < -2 * EPS_CSV).any():
        raise ValueError("Prefix duration exceeds full duration")
    dt = np.maximum(dt, 0)
    # Preserve the input-consistency check without materializing unused IAT columns.
    residual_moments(full_n - 1, f.mean_iat, f.std_iat, pre_n - 1, p.mean_iat, p.std_iat, dt)
    frame["r_byte_up_share"] = divide(remaining["up"], full_b - pre_b)
    frame["r_after5_seconds"] = dt
    frame["p_byte_share"] = divide(pre_b, full_b)
    return frame


def make_selected_frame(task, raw, prefix=None):
    if task != TASKS[0] or prefix is None:
        raise ValueError("Task 1 requires the matching first-five-packet rows")
    frame = paired_frame(raw, prefix)
    keys = size_ops.fingerprint_keys(frame)
    for name in keys:
        frame[name] = keys[name].to_numpy()
    frame["transport"] = BinaryTransportEncoder().transform(frame[["transport"]]).ravel().astype(int)
    return frame


def prefix_frame(raw):
    """Build the fixed 33-column Task 2 content design from prefix statistics."""
    if not (raw.packet_count_up + raw.packet_count_down).eq(5).all():
        raise ValueError("Task 2 requires exactly five packets")
    bounds = prefix_size_bounds(raw)
    frame = raw.copy()
    keys = size_ops.task2_keys(raw, bounds.p_up_max_upper.to_numpy())
    for name in keys:
        frame[name] = keys[name].to_numpy()
    frame["transport"] = BinaryTransportEncoder().transform(raw[["transport"]]).ravel().astype(int)
    for name in bounds:
        frame[name] = bounds[name].to_numpy()
    return frame


class BinaryTransportEncoder(TransformerMixin, BaseEstimator):
    """Fixed mapping, including when a fitting fold contains only one protocol.

    Accept original strings or already encoded 0/1. Missing/unknown protocols
    raise an error rather than silently representing them as TCP.
    """

    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=object)
        if values.ndim != 2 or values.shape[1] != 1:
            raise ValueError("Expected exactly one transport column")
        codes = []
        for value in values[:, 0]:
            if isinstance(value, str):
                if value not in TRANSPORT_CODES:
                    raise ValueError(f"Unknown transport: {value!r}")
                codes.append(TRANSPORT_CODES[value])
            elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
                codes.append(int(value))
            else:
                raise ValueError(f"Missing or invalid transport: {value!r}")
        return np.asarray(codes, dtype=float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["transport"], dtype=object)


def log_preprocessor(preprocess):
    pre = clone(preprocess)
    branches = list(pre.transformers)
    name, _, columns = branches[0]
    steps = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("log", FunctionTransformer(signed_log1p)),
    ]
    branches[0] = (name, Pipeline(steps), list(columns))
    return pre.set_params(transformers=branches)


def native(pre, frame, encoded):
    width = len(pre.transformers_[0][2])
    cats = list(pre.transformers_[2][2])
    result = pd.DataFrame(
        encoded[:, : width + 1], columns=[f"numeric_{i}" for i in range(width)] + ["transport"]
    )
    for key in cats:
        result[key] = frame[key].astype(str).to_numpy()
    return (result, cats)


def component_probabilities(component, frame):
    work = frame[component["columns"]]
    a = component["preprocess"].transform(work)
    if component["kind"] in ("cat", "full_cat"):
        x, cats = native(component["preprocess"], work, a)
        assert cats == component["cats"]
    else:
        assert component["kind"] == "et"
        x = a
    model = component["model"]
    raw = model.predict_proba(x)
    p = np.zeros((len(frame), 6))
    for j, c in enumerate(model.classes_):
        p[:, CLASSES.index(str(c))] = raw[:, j]
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-06, rtol=1e-05)
    return p


def structural(frame, catalog):
    vocab = size_ops.vocabulary(catalog, np.arange(len(catalog)))
    return structural_query(frame, catalog, vocab), dict(vocab)


def structural_query(frame, catalog, vocabulary):
    keys = size_ops.structural_keys(catalog, vocabulary)
    work = frame.copy()
    for col in keys:
        work[col] = keys[col].to_numpy()
    return work
