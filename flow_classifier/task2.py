"""Task 2: original-coordinate ET, CatBoost, size density and supported fusion.

No PCA axes or rotated trees are constructed, stored, or evaluated.
"""

import copy
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.base import clone
from . import CLASSES, TASKS, fold_indices
from . import features as feature_ops
from . import sizes as size_ops
from . import support as support_ops


def fusion_objective(theta, x, target, strength):
    logits = x @ theta[:3] + theta[None, 3:]
    logs = logits - logsumexp(logits, axis=1)[:, None]
    residual = (np.exp(logs) - target) / len(target)
    base_delta = np.r_[theta[:2] - 0.5, theta[3:]]
    loss = float(
        -(target * logs).sum() / len(target)
        + 0.05 * (base_delta @ base_delta)
        + 0.5 * strength * theta[2] ** 2
    )
    gradient = np.r_[np.einsum("nc,nck->k", residual, x), residual.sum(0)]
    gradient[:2] += 0.1 * (theta[:2] - 0.5)
    gradient[2] += strength * theta[2]
    gradient[3:] += 0.1 * theta[3:]
    return (loss, gradient)


def three_source_values(et, cat, size):
    return np.log(np.clip(np.stack((et, cat, size), axis=2), 1e-07, 1.0))


def fit_fusion(et, cat, size, labels, strength):
    """Fit one objective, first constraining the size weight to zero."""
    x = three_source_values(et, cat, size)
    target = (np.asarray(labels)[:, None] == np.asarray(CLASSES)[None, :]).astype(float)
    if strength <= 0 or not (target.sum(1) == 1).all():
        raise ValueError("Invalid fusion labels or regularization")
    theta = np.r_[0.5, 0.5, 0.0, np.zeros(6)]
    free_offsets = [(None, None)] * 6
    stages = [([(0, None)] * 2 + [(0, 0)] + free_offsets, 20),
              ([(0, None)] * 3 + free_offsets, 50)]
    for bounds, maxls in stages:
        solved = minimize(fusion_objective, theta, args=(x, target, strength), jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options=dict(maxiter=2000, ftol=1e-12, gtol=1e-8, maxls=maxls))
        _, gradient = fusion_objective(solved.x, x, target, strength)
        for j, (lower, upper) in enumerate(bounds):
            if lower is not None and (lower == upper or (solved.x[j] < lower + 1e-9 and gradient[j] > 0)):
                gradient[j] = 0
        if not solved.success and abs(gradient).max() >= 1e-5:
            raise RuntimeError(f"Fusion fitting failed: {solved.message}")
        theta = solved.x
    return dict(theta=theta)


def fusion_scores(model, et, cat, size):
    return three_source_values(et, cat, size) @ model["theta"][:3] + model["theta"][None, 3:]


def fit_size_density(prefix, labels, sigma):
    model = size_ops.fit_density(
        size_ops.multiset_catalog(prefix), np.asarray(labels), prefix.transport.to_numpy(), sigma, 10.0
    )
    return dict(
        model=model,
        fit_ids=prefix.flow_id.tolist(),
        fit_labels=np.asarray(labels),
        sigma=sigma,
        alpha=10.0,
    )


def size_probabilities(latent, prefix):
    score, available = size_ops.density_scores(
        latent["model"], size_ops.multiset_catalog(prefix), prefix.transport.to_numpy()
    )
    p = softmax(score, axis=1)
    p[available == 0] = 1 / 6
    return (p, available)


def apply(model, raw, et, candidate):
    default = np.asarray(model["classes"])[et.argmax(1)]
    return support_ops.limit_protocol_changes(model, raw, default, candidate)


def source_probabilities(head, raw):
    base = head["base"]
    frame = feature_ops.prefix_frame(raw)
    encoded = base["pipeline"][:-1].transform(frame[base["columns"]])
    et = base["pipeline"].named_steps["model"].predict_proba(encoded)
    matrix, keys = feature_ops.native(base["pipeline"].named_steps["preprocess"], frame, encoded)
    assert keys == base["catboost_keys"]
    return et, base["catboost"].predict_proba(matrix), size_probabilities(head["latent"], raw)[0]


def predict(bundle, raw):
    source = bundle["source"]
    ep, cat, size = source_probabilities(source, raw)
    quality = fusion_scores(bundle["fusion"], ep, cat, size)
    root = np.asarray(CLASSES)[quality.argmax(1)]
    root = support_ops.adjust_time(quality, root, bundle["time"], raw, .5)
    safe = apply(source["support"], raw, ep, root)
    middle = support_ops.predict_size(source["size"], raw, safe)
    return support_ops.select_past_capture(source["capture"], raw, middle)


def fit_head(recipe, raw, y):
    frame = feature_ops.prefix_frame(raw)
    pipe = clone(recipe["pipeline"]).fit(frame[recipe["columns"]], y)
    np.testing.assert_array_equal(pipe.named_steps["model"].classes_, CLASSES)
    encoded = pipe[:-1].transform(frame[recipe["columns"]])
    matrix, keys = feature_ops.native(pipe.named_steps["preprocess"], frame, encoded)
    cat = type(recipe["catboost"])(**copy.deepcopy(recipe["catboost"].get_params())).fit(matrix, y)
    return dict(base=dict(pipeline=pipe, columns=recipe["columns"], catboost=cat, catboost_keys=keys),
                latent=fit_size_density(raw, y, 1.0))


def fit(recipe, raw, y, scope):
    probabilities = [np.full((len(raw), 6), np.nan) for _ in range(3)]
    seen = []
    for i, entry in enumerate(recipe["memberships"][scope]):
        fit, held = fold_indices(raw, entry)
        head = fit_head(recipe, raw.iloc[fit], y[fit])
        for output, values in zip(probabilities, source_probabilities(head, raw.iloc[held]), strict=True):
            output[held] = values
        seen += entry["held_ids"]
        print("cold Task2", scope, "inner", i+1, "/3", flush=True)
    assert len(seen) == len(set(seen)) == len(raw) and set(seen) == set(raw.flow_id)
    assert all(np.isfinite(p).all() for p in probabilities)
    fusion = fit_fusion(*probabilities, y, .1)
    source = fit_head(recipe, raw, y)
    source.update(size=support_ops.fit_size(raw, y, recipe["size"]),
                  capture=support_ops.fit_past_capture(raw, y),
                  support=support_ops.fit_protocol_guard(raw, y, minimum_rows=2),
                  fit_ids=raw.flow_id.tolist(), fit_labels=y)
    return dict(source=source, fusion=fusion,
                time=support_ops.fit_time(raw, y),
                fit_ids=raw.flow_id.tolist(), fit_labels=y, predictor="flow_classifier.task2.predict")
