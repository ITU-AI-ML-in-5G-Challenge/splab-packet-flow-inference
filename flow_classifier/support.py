"""Size, capture and temporal support; fitted only on labeled training rows."""

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from scipy.special import expit
from scipy.special import logit
from . import CLASSES
from . import features as feature_ops
from . import sizes as size_ops

SIZE_TOLERANCE = 1e-06


def fit_protocol_guard(raw, labels, minimum_rows):
    """Count fitting examples for each class/protocol; no query labels are used."""
    if not isinstance(minimum_rows, int) or minimum_rows < 1:
        raise ValueError("minimum_rows must be a positive integer")
    labels = np.asarray(labels)
    if len(labels) != len(raw):
        raise ValueError("Fitting rows and labels must be aligned")
    counts = np.array(
        [[np.sum((labels == c) & raw.transport.eq(p).to_numpy()) for p in ("tcp", "udp")] for c in CLASSES],
        dtype=int,
    )
    return dict(classes=CLASSES, counts=counts, minimum_rows=minimum_rows, fit_ids=raw.flow_id.tolist())


def limit_protocol_changes(model, raw, default, proposed):
    """Keep the existing default if either side lacks class/protocol support."""
    classes = {c: i for i, c in enumerate(model["classes"])}
    protocol = raw.transport.map({"tcp": 0, "udp": 1}).to_numpy(int)
    old = np.array([classes[c] for c in default])
    new = np.array([classes[c] for c in proposed])
    insufficient = (model["counts"][old, protocol] < model["minimum_rows"]) | (
        model["counts"][new, protocol] < model["minimum_rows"]
    )
    return np.where(insufficient, default, proposed)


def fit_size(raw, labels, config):
    x = raw[["mean_pkt_size_down", "std_pkt_size_down"]].to_numpy(float)
    labels = np.asarray(labels)
    valid = np.isfinite(x).all(1) & raw.packet_count_down.gt(0).to_numpy()
    groups = {}
    for i, key in enumerate(feature_ops.keys(raw)):
        if valid[i]:
            groups.setdefault(key, []).append(i)
    models = {}
    for key, ids in groups.items():
        ids = np.array(ids)
        if len(set(labels[ids])) < 2:
            continue
        points, inverse = np.unique(x[ids], axis=0, return_inverse=True)
        if len(points) < config["minimum_points"]:
            continue
        target = np.zeros((len(points), 6))
        ambiguous = np.zeros(len(points), bool)
        for j in range(len(points)):
            present = labels[ids][inverse == j]
            ambiguous[j] = len(set(present)) > 1
            target[j] = [(present == c).mean() for c in CLASSES]
        center = points.mean(0)
        _, _, v = np.linalg.svd(points - center, full_matrices=False)
        axis = v[0]
        coordinate = (points - center) @ axis
        residual = points - center - coordinate[:, None] * axis
        if np.max(np.linalg.norm(residual, axis=1)) <= SIZE_TOLERANCE:
            order = np.argsort(coordinate)
            entry = dict(
                dimension=1,
                center=center,
                axis=axis,
                coordinate=coordinate[order],
                points=points[order],
                target=target[order],
                ambiguous=ambiguous[order],
            )
        else:
            if len(points) < 3:
                continue
            try:
                tri = Delaunay(points)
            except QhullError:
                continue
            entry = dict(dimension=2, triangulation=tri, points=points, target=target, ambiguous=ambiguous)
        models[key] = entry
    return dict(config=config, groups=models, fit_ids=raw.flow_id.tolist())


def predict_size(model, raw, reference):
    x = raw[["mean_pkt_size_down", "std_pkt_size_down"]].to_numpy(float)
    output = np.asarray(reference).copy()
    for i, key in enumerate(feature_ops.keys(raw)):
        entry = model["groups"].get(key)
        if entry is None or not np.isfinite(x[i]).all() or raw.packet_count_down.iloc[i] <= 0:
            continue
        point = entry["points"]
        distance = np.square(point - x[i]).sum(1)
        near = int(np.argmin(distance))
        if distance[near] <= SIZE_TOLERANCE**2 and entry["ambiguous"][near]:
            continue
        if entry["dimension"] == 1:
            z = (x[i] - entry["center"]) @ entry["axis"]
            orth = x[i] - entry["center"] - z * entry["axis"]
            t = entry["coordinate"]
            if (
                np.linalg.norm(orth) > SIZE_TOLERANCE
                or z < t[0] - SIZE_TOLERANCE
                or z > t[-1] + SIZE_TOLERANCE
            ):
                continue
            z = np.clip(z, t[0], t[-1])
            right = min(max(np.searchsorted(t, z, side="right"), 1), len(t) - 1)
            left = right - 1
            portion = (z - t[left]) / (t[right] - t[left])
            ids = np.array([left, right])
            w = np.array([1 - portion, portion])
        else:
            tri = entry["triangulation"]
            simplex = int(tri.find_simplex(x[i], tol=1e-12))
            if simplex < 0:
                continue
            transform = tri.transform[simplex]
            b = transform[:2] @ (x[i] - transform[2])
            w = np.r_[b, 1 - b.sum()]
            ids = tri.simplices[simplex]
            if (w < -1e-10).any():
                continue
            w = np.maximum(w, 0)
            w /= w.sum()
        if entry["ambiguous"][ids[w > 1e-10]].any():
            continue
        p = w @ entry["target"][ids]
        if p.max() <= 0.5 + 1e-10:
            continue
        label = CLASSES[int(p.argmax())]
        output[i] = label
    return output


def size_signatures(raw):
    result = []
    for row in raw.itertuples():
        key = [row.transport]
        for direction in ("up", "down"):
            key.extend(size_ops.moments(getattr(row, "packet_count_" + direction),
                                       getattr(row, "bytes_" + direction),
                                       getattr(row, "mean_pkt_size_" + direction),
                                       getattr(row, "std_pkt_size_" + direction)))
        result.append(tuple(key))
    return result


def signature_profile(key):
    transport, up_count, _, _, down_count, _, _ = key
    return transport, up_count, down_count


def capture_features(raw, origin):
    gaps = feature_ops.coordinates(raw)
    time = (raw.start_rel.to_numpy(float) - origin) / 120
    sizes = np.log1p(raw[["bytes_up", "bytes_down", "std_pkt_size_up", "std_pkt_size_down"]].to_numpy(float))
    return np.column_stack([gaps, time, sizes])


def fit_capture_expert(raw, labels, cache=None, *, keys=None):
    y = np.asarray(labels)
    if keys is None:
        keys = size_signatures(raw)
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    entries = {}
    for key, ids in groups.items():
        active = [c for c in CLASSES if np.sum(y[ids] == c) >= 3]
        if len(active) < 2:
            continue
        pool = np.array([i for i, k in enumerate(keys) if signature_profile(k) == signature_profile(key) and y[i] in active])
        fit_ids = raw.flow_id.iloc[pool].tolist()
        signature = tuple(fit_ids)
        entry = cache.get(signature) if cache is not None else None
        if entry is None:
            origin = float(raw.start_rel.iloc[pool].min())
            learner = RandomForestClassifier(n_estimators=100, max_features=1.0, min_samples_leaf=2,
                                            class_weight="balanced", random_state=9401, n_jobs=1).fit(
                                                capture_features(raw.iloc[pool], origin), y[pool])
            entry = dict(learner=learner, active=active, origin=origin, fit_ids=fit_ids)
            if cache is not None:
                cache[signature] = entry
        entries[key] = entry
    return dict(entries=entries)


def fit_past_capture(raw, labels):
    """Share each fitted RF and its provenance across time snapshots."""
    labels = np.asarray(labels)
    starts = raw.start_rel.to_numpy(float)
    times = np.unique(starts)
    keys = size_signatures(raw)
    cache, snapshots = {}, []
    for t in times:
        mask = starts <= t
        prefix_keys = [keys[i] for i in np.flatnonzero(mask)]
        expert = fit_capture_expert(raw.loc[mask], labels[mask], cache, keys=prefix_keys)
        snapshots.append(dict(experts={"strict": expert}))
    return dict(times=times, snapshots=snapshots, fit_ids=raw.flow_id.tolist(),
                reference_policy="strictly_earlier_start", distinct_forests=len(cache))


def select_past_capture(model, raw, reference):
    """Use only the last labeled prefix whose final start is < query start."""
    cuts = np.searchsorted(model["times"], raw.start_rel.to_numpy(float), side="left") - 1
    output = np.asarray(reference).copy()
    for cut in np.unique(cuts):
        if cut < 0:
            continue
        ids = np.flatnonzero(cuts == cut)
        assert (model["times"][cut] < raw.start_rel.iloc[ids]).all()
        output[ids] = select_capture(
            model["snapshots"][cut],
            raw.iloc[ids],
            np.asarray(reference)[ids],
        )
    return output


def select_capture(model, raw, reference):
    """Adopt the RF argmax within the fitted signature/class support."""
    default = np.asarray(reference)
    out = default.copy()
    expert = model["experts"]["strict"]
    groups = defaultdict(list)
    for i, key in enumerate(size_signatures(raw)):
        entry = expert["entries"].get(key)
        if entry is not None and default[i] in entry["active"]:
            groups[key].append(i)
    for key, rows in groups.items():
        entry = expert["entries"][key]
        ids = np.asarray(rows, dtype=int)
        x = capture_features(raw.iloc[ids], entry["origin"])
        probability = entry["learner"].predict_proba(x)
        full = np.zeros((len(ids), 6))
        for j, c in enumerate(entry["learner"].classes_):
            full[:, CLASSES.index(c)] = probability[:, j]
        out[ids] = np.asarray(CLASSES)[full.argmax(1)]
    return out


TIME_TOLERANCE = 64 * np.finfo(float).eps


def time_emissions(background, noise):
    return (1 - noise[:, None, None]) * np.eye(6)[None, :, :] + noise[:, None, None] * background[:, None, :]


def protocol_codes(raw):
    v = raw.transport.map({"tcp": 0, "udp": 1})
    assert v.notna().all()
    return v.to_numpy(int)


def time_context(model, raw):
    q = raw.start_rel.to_numpy(float) - model["origin"]
    proto = protocol_codes(raw)
    times = model["times"]
    prior = model["prior"]
    result = []
    assert np.isfinite(q).all()
    for t, p in zip(q, proto):
        # Simultaneous starts are not earlier flows and must be excluded.
        left = np.searchsorted(times, t, "left") - 1
        tau = model["tau"]
        if left < 0:
            past = prior.copy()
        else:
            ratio = (t - times[left]) / tau
            a = np.exp(-ratio)
            past = a * model["forward"][left] + -np.expm1(-ratio) * prior
        posterior = past
        posterior /= posterior.sum()
        result.append(posterior @ model["query_emission"][p])
    return np.asarray(result).reshape(-1, 6)


def time_objective(theta, times, counts, prior, background, messages=False):
    tau = np.exp(theta[0])
    noise = expit(theta[1:])
    e = time_emissions(background, noise)
    logs = np.einsum("ipc,pzc->iz", counts, np.log(e))
    offset = logs.max(1)
    scaled = np.exp(logs - offset[:, None])
    derivative = np.array(
        [
            counts[:, p] @ ((background[p][None, :] - np.eye(6)) / e[p] * noise[p] * (1 - noise[p])).T
            for p in range(2)
        ]
    )
    alpha = prior.copy()
    sensitivity = np.zeros((3, 6))
    ll = 0.0
    gradient = np.zeros(3)
    forward = []
    for i in range(len(times)):
        if i:
            ratio = (times[i] - times[i - 1]) / tau
            a = np.exp(-ratio)
            b = -np.expm1(-ratio)
            pred = a * alpha + b * prior
            ds = a * sensitivity
            ds[0] += a * ratio * (alpha - prior)
        else:
            pred = prior.copy()
            ds = np.zeros((3, 6))
        value = pred * scaled[i]
        dv = ds * scaled[i]
        dv[1:] += value[None, :] * derivative[:, i, :]
        normal = value.sum()
        assert normal > 0 and np.isfinite(normal)
        dn = dv.sum(1)
        alpha = value / normal
        sensitivity = (dv - alpha[None, :] * dn[:, None]) / normal
        ll += offset[i] + np.log(normal)
        gradient += dn / normal
        forward.append(alpha.copy())
    n = counts.sum()
    loss = float(-ll / n)
    g = -gradient / n
    return (loss, g, np.asarray(forward), scaled) if messages else (loss, g)


def time_evidence(model, raw):
    p = time_context(model, raw)
    prior = model["query_prior"][protocol_codes(raw)]
    r = np.log1p((p - prior) / prior)
    r[abs(r) <= TIME_TOLERANCE] = 0.0
    assert np.isfinite(r).all()
    return r


def prepare_time(times, counts, prior, background, tau, noise):
    if np.all(noise == 1):
        forward = np.tile(prior, (len(times), 1))
        scaled = np.ones_like(forward)
    else:
        _, _, forward, scaled = time_objective(
            np.r_[np.log(tau), logit(noise)], times, counts, prior, background, True
        )
    e = time_emissions(background, noise)
    unconditional = np.einsum("z,pzc->pc", prior, e)
    return dict(
        times=times,
        prior=prior,
        tau=float(tau),
        forward=forward,
        query_emission=e,
        query_prior=unconditional,
    )


def adjust_time(scores, current, time, raw, strength=0.5, eligible=None):
    z = np.asarray(scores) + strength * time_evidence(time, raw)
    if eligible is not None:
        z = np.where(eligible, z, -np.inf)
    ix = np.arange(len(raw))
    old = np.array([CLASSES.index(str(c)) for c in current])
    candidate = z.argmax(1)
    tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(z[ix, candidate]), abs(z[ix, old])))
    return np.where(z[ix, candidate] > z[ix, old] + tolerance, np.asarray(CLASSES)[candidate], current)


def fit_time(raw, labels):
    labels = np.asarray(labels)
    original = raw.start_rel.to_numpy(float)
    proto = protocol_codes(raw)
    assert len(raw) == len(labels) > 0 and np.isfinite(original).all()
    origin = float(original.min())
    times, inverse = np.unique(original - origin, return_inverse=True)
    counts = np.zeros((len(times), 2, 6), int)
    indices = np.array([CLASSES.index(str(c)) for c in labels])
    np.add.at(counts, (inverse, proto, indices), 1)
    totals = counts.sum(0)
    prior = (totals.sum(0) + 1) / (len(raw) + 6)
    background = (totals + 1) / (totals.sum(1)[:, None] + 6)
    positive = np.diff(times)
    positive = positive[positive > 0]
    span = float(np.ptp(times))
    scale = max(span, float(np.median(positive)) if len(positive) else 1.0, 1e-09)
    bounds = [(np.log(max(scale * 1e-12, 1e-09)), np.log(scale * 1000000.0))] + [
        (float(logit(1e-06)), float(logit(1 - 1e-06)))
    ] * 2
    null = float(-np.sum(totals * np.log(background)) / len(raw))
    best = dict(loss=null, tau=scale, noise=np.ones(2), converged=True, projected_gradient=0.0)
    starts = sorted(set([float(np.median(positive)) if len(positive) else 1.0, max(span / 6, 1e-09)]))
    for tau in starts:
        for noise in ((0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1)):
            initial = np.r_[np.clip(np.log(tau), *bounds[0]), logit(noise)]
            result = minimize(
                time_objective,
                initial,
                args=(times, counts, prior, background),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options=dict(maxiter=1000, ftol=1e-12, gtol=1e-08, maxls=50),
            )
            loss, g = time_objective(result.x, times, counts, prior, background)
            for j, (lo, hi) in enumerate(bounds):
                if result.x[j] < lo + 1e-07 and g[j] > 0:
                    g[j] = 0
                if result.x[j] > hi - 1e-07 and g[j] < 0:
                    g[j] = 0
            record = dict(
                loss=loss,
                tau=float(np.exp(result.x[0])),
                noise=expit(result.x[1:]),
                converged=bool(result.success),
                projected_gradient=float(abs(g).max()),
            )
            if loss < best["loss"] - 1e-12:
                best = record
    assert best["converged"] or best["projected_gradient"] < 1e-05
    m = prepare_time(times, counts, prior, background, best["tau"], best["noise"])
    m.update(
        origin=origin,
        fit_ids=raw.flow_id.tolist(),
        fit_labels=labels,
        fit_times=original,
        fit_transport=raw.transport.to_numpy(),
        reference_policy="strictly_earlier_start",
    )
    return m
