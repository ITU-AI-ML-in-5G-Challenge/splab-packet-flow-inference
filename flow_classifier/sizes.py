"""Integer moment constraints, fingerprint vocabulary and latent size density."""

from math import isqrt
import numpy as np
from collections import Counter
from functools import lru_cache
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import gammaln
from scipy.special import logsumexp
from . import CLASSES

def _roots(first, second, S, Q, lower):
    rest_sum = S - first - second
    rest_square = Q - first * first - second * second
    disc = 2 * rest_square - rest_sum * rest_sum
    mask = disc >= 0
    if not mask.any():
        return np.empty((0, 4), dtype=np.int64)
    first, second, rest_sum, disc = (x[mask] for x in (first, second, rest_sum, disc))
    root = np.floor(np.sqrt(disc)).astype(np.int64)
    mask = (root * root == disc) & ((rest_sum - root) % 2 == 0)
    if not mask.any():
        return np.empty((0, 4), dtype=np.int64)
    first, second, rest_sum, root = (x[mask] for x in (first, second, rest_sum, root))
    third = (rest_sum - root) // 2
    fourth = (rest_sum + root) // 2
    mask = (third >= second) & (fourth >= third) & (first >= lower)
    return np.column_stack((first[mask], second[mask], third[mask], fourth[mask]))


def _two(S, Q, lower):
    disc = 2 * Q - S * S
    if disc < 0:
        return np.empty((0, 2), dtype=np.int64)
    root = isqrt(disc)
    if root * root != disc or (S - root) % 2:
        return np.empty((0, 2), dtype=np.int64)
    a, b = ((S - root) // 2, (S + root) // 2)
    return np.asarray([(a, b)], dtype=np.int64) if a >= lower else np.empty((0, 2), dtype=np.int64)


def feasible_multisets(n, S, Q, lower=0):
    """All sorted integer solutions for 0<=n<=4; n=5 is not enumerated."""
    n, S, Q, lower = map(int, (n, S, Q, lower))
    if n < 0 or n > 5 or S < 0 or (Q < 0) or (lower < 0):
        raise ValueError("Invalid packet moments")
    if n == 5:
        return None
    if n == 0:
        return np.empty((1, 0), dtype=np.int64) if S == Q == 0 else np.empty((0, 0), dtype=np.int64)
    if n == 1:
        return (
            np.asarray([[S]], dtype=np.int64)
            if S >= lower and Q == S * S
            else np.empty((0, 1), dtype=np.int64)
        )
    if n == 2:
        return _two(S, Q, lower)
    if n == 3:
        a = np.arange(lower, S // 3 + 1, dtype=np.int64)
        rest_sum = S - a
        rest_square = Q - a * a
        disc = 2 * rest_square - rest_sum * rest_sum
        mask = disc >= 0
        if not mask.any():
            return np.empty((0, 3), dtype=np.int64)
        a, rest_sum, disc = (a[mask], rest_sum[mask], disc[mask])
        root = np.floor(np.sqrt(disc)).astype(np.int64)
        mask = (root * root == disc) & ((rest_sum - root) % 2 == 0)
        a, rest_sum, root = (a[mask], rest_sum[mask], root[mask])
        b = (rest_sum - root) // 2
        c = (rest_sum + root) // 2
        mask = (b >= a) & (c >= b)
        return np.column_stack((a[mask], b[mask], c[mask]))
    chunks = []
    for aval in range(lower, S // 4 + 1):
        b = np.arange(aval, (S - aval) // 3 + 1, dtype=np.int64)
        if not len(b):
            continue
        a = np.full(len(b), aval, dtype=np.int64)
        found = _roots(a, b, S, Q, lower)
        if len(found):
            chunks.append(found)
    return np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.int64)


KEYS = ("fp_up_bytes", "fp_up_max_bound", "fp_down_bytes")
TM = "struct_tcp_m"
UNIQUE = "struct_up_unique"
VOCAB = "struct_up_vocab"


def moments(n, total, mean, sd):
    if not np.isfinite([n, total, mean, sd]).all() or sd < 0:
        raise ValueError("Invalid moments")
    N, S = (int(round(n)), int(round(total)))
    q = N * (mean * mean + sd * sd)
    Q = int(round(q))
    if abs(n - N) > 1e-08 or abs(total - S) > 1e-08 or abs(N * mean - S) > 1e-06 or (abs(q - Q) > 0.0001):
        raise ValueError("Moments are not consistent with integer sizes")
    if not 0 <= N <= 5:
        raise ValueError("Prefix count outside 0..5")
    if N == 0 and (S != 0 or Q != 0):
        raise ValueError("Nonzero absent-direction moments")
    return (N, S, Q)


def size_key(proto, values):
    if values is None or len(values) == 0:
        return proto + "|NA|NA"
    mid = str(int(values[len(values) // 2])) if len(values) % 2 == 1 and len(values) >= 3 else "NA"
    return proto + "|" + str(int(values[-1])) + "|" + mid


def vocabulary(cat, fit):
    counts = Counter()
    for index in fit:
        for dr in ("up", "down"):
            a = cat[index][dr]["solutions"]
            if a is not None and len(a) == 1:
                counts.update((int(v) for v in a[0]))
    return counts


def fingerprint_keys(frame):
    """Protocol/count/byte composites; maximum bound is never an actual max claim."""
    keys = pd.DataFrame(index=frame.index)
    proto = frame.transport.astype(str)
    count = frame.p_packet_count_up.astype(int).astype(str)
    for source, name in (("p_bytes_up", "fp_up_bytes"), ("p_bytes_down", "fp_down_bytes")):
        a = frame[source].to_numpy(float)
        if not np.isfinite(a).all() or not np.allclose(a, np.rint(a), atol=1e-07, rtol=0):
            raise ValueError("Packet byte totals must be integers")
        keys[name] = (
            proto + "|" + count + "|" + pd.Series(np.rint(a).astype(int), index=frame.index).astype(str)
        )
    maximum = frame.p_up_max_upper.to_numpy(float)
    rounded = ["NA" if not np.isfinite(x) else str(int(np.rint(x))) for x in maximum]
    keys["fp_up_max_bound"] = proto + "|" + pd.Series(rounded, index=frame.index)
    return keys[list(KEYS)]


def structural_keys(catalog, vocab):
    """Encode size candidates using a vocabulary fitted on reference rows."""
    rows = []
    for item in catalog:
        a = item["up"]["solutions"]
        proto = item["transport"]
        unique = a[0] if a is not None and len(a) == 1 else None
        chosen = None
        if a is not None and len(a):
            chosen = max(a, key=lambda values: sum(np.log1p(vocab.get(int(v), 0)) for v in values))
        rows.append(
            {
                TM: proto + "|" + ("NA" if item["tm"] is None else str(item["tm"])),
                UNIQUE: size_key(proto, unique),
                VOCAB: size_key(proto, chosen),
            }
        )
    return pd.DataFrame(rows, columns=[TM, UNIQUE, VOCAB])


@lru_cache(maxsize=4096)
def multiset_solutions(n, total, square):
    if n == 5:
        out = (
            np.full((1, n), total // n, dtype=np.int64)
            if n * square == total * total and total % n == 0 and (total > 0)
            else None
        )
    else:
        out = feasible_multisets(n, total, square, lower=1)
    if out is not None:
        assert len(out) > 0
        np.testing.assert_array_equal(out.sum(1), total)
        np.testing.assert_array_equal((out * out).sum(1), square)
        assert not n or (out > 0).all()
        out.flags.writeable = False
    return out


@lru_cache(maxsize=4096)
def structural_solutions(n, S, Q):
    if n == 0:
        a = np.empty((1, 0), dtype=np.int64)
    elif n <= 3:
        a = feasible_multisets(n, S, Q, lower=1)
    elif n * Q == S * S and S % n == 0 and (S // n > 0):
        a = np.full((1, n), S // n, dtype=np.int64)
    else:
        return None
    if not len(a):
        raise ValueError("No positive integer solution")
    a.flags.writeable = False
    return a


def task2_keys(raw, max_upper):
    n = raw.packet_count_up.to_numpy(float)
    if not (raw.packet_count_up + raw.packet_count_down == 5).all():
        raise ValueError("Task 2 requires five-packet prefixes")
    mean = raw.mean_pkt_size_up.to_numpy(float)
    std = raw.std_pkt_size_up.to_numpy(float)
    if not np.isfinite(n + mean + std).all() or (n < 0).any() or (std < 0).any():
        raise ValueError("Invalid prefix moments")
    frame = pd.DataFrame(
        dict(
            transport=raw.transport.to_numpy(),
            p_packet_count_up=n,
            p_bytes_up=raw.bytes_up.to_numpy(),
            p_bytes_down=raw.bytes_down.to_numpy(),
            p_up_max_upper=max_upper,
        )
    )
    return fingerprint_keys(frame)


def multiset_catalog(raw):
    assert (raw.packet_count_up + raw.packet_count_down).eq(5).all()
    items = []
    for row in raw.itertuples(index=False):
        item = {}
        for direction in ("up", "down"):
            values = moments(
                getattr(row, "packet_count_" + direction),
                getattr(row, "bytes_" + direction),
                getattr(row, "mean_pkt_size_" + direction),
                getattr(row, "std_pkt_size_" + direction),
            )
            item[direction] = multiset_solutions(*values)
        items.append(item)
    return items


def structural_catalog(raw):
    if not (raw.packet_count_up + raw.packet_count_down).eq(5).all():
        raise ValueError("Prefix must contain five packets")
    rows = []
    for row in raw.itertuples(index=False):
        proto = row.transport
        if proto not in ("tcp", "udp"):
            raise ValueError("Invalid transport")
        item = {"transport": proto}
        for dr in ("up", "down"):
            n, S, Q = moments(
                getattr(row, "packet_count_" + dr),
                getattr(row, "bytes_" + dr),
                getattr(row, "mean_pkt_size_" + dr),
                getattr(row, "std_pkt_size_" + dr),
            )
            item[dr] = {"n": n, "sum": S, "square": Q, "solutions": structural_solutions(n, S, Q)}
        n, S, Q = (item["up"][k] for k in ("n", "sum", "square"))
        tm = None
        if proto == "tcp" and n >= 3:
            count_full = n - 3
            s = S - 66 - count_full * 1454
            q = Q - 66**2 - count_full * 1454**2
            if s > 0 and q > 0:
                pair = feasible_multisets(2, s, q, lower=1)
                if len(pair) == 1:
                    tm = int(pair[0, 0])
        item["tm"] = tm
        rows.append(item)
    return rows


ITERATIONS = 40


def basis(sizes, centers, sigma, geometric_mean):
    sizes = np.asarray(sizes, int)
    centers = np.asarray(centers, int)
    radius = int(np.ceil(8 * sigma))
    rows = []
    cols = []
    values = []
    for j, c in enumerate(centers):
        grid = np.arange(max(1, c - radius), c + radius + 1)
        z = np.exp(-0.5 * ((grid - c) / sigma) ** 2).sum()
        valid = np.flatnonzero(abs(sizes - c) <= radius)
        rows.extend(valid)
        cols.extend([j] * len(valid))
        values.extend(np.exp(-0.5 * ((sizes[valid] - c) / sigma) ** 2) / z)
    probability = 1.0 / max(geometric_mean, 1.000001)
    rows.extend(range(len(sizes)))
    cols.extend([len(centers)] * len(sizes))
    values.extend(probability * np.exp((sizes - 1) * np.log1p(-probability)))
    return csr_matrix((values, (rows, cols)), shape=(len(sizes), len(centers) + 1))


def likelihood(rep, matrix, theta):
    density = np.asarray(matrix @ theta).ravel()
    assert np.isfinite(density).all() and (density > 0).all()
    candidate = np.asarray(rep["counts"] @ np.log(density)).ravel() + rep["permutation"]
    bag = np.array([logsumexp(candidate[s]) for s in rep["slices"]])
    return (density, candidate, bag)


def representation(bags):
    sizes = np.unique(np.concatenate([a.ravel() for a in bags])) if bags else np.empty(0, int)
    index = {int(s): j for j, s in enumerate(sizes)}
    rows = []
    cols = []
    values = []
    permutation = []
    slices = []
    position = 0
    for a in bags:
        slices.append(slice(position, position + len(a)))
        for candidate in a:
            unique, count = np.unique(candidate, return_counts=True)
            rows.extend([position] * len(unique))
            cols.extend((index[int(s)] for s in unique))
            values.extend(count)
            permutation.append(float(gammaln(len(candidate) + 1) - gammaln(count + 1).sum()))
            position += 1
    return dict(
        sizes=sizes,
        counts=csr_matrix((values, (rows, cols)), shape=(position, len(sizes))),
        permutation=np.asarray(permutation),
        slices=slices,
    )


def optimize(rep, matrix, prior, alpha):
    theta = prior.copy()
    for _ in range(ITERATIONS):
        density, candidate, bag = likelihood(rep, matrix, theta)
        posterior = np.empty(len(candidate))
        for group, value in zip(rep["slices"], bag):
            posterior[group] = np.exp(candidate[group] - value)
        expected_sizes = np.asarray(rep["counts"].T @ posterior).ravel()
        expected_components = theta * np.asarray(matrix.T @ (expected_sizes / density)).ravel()
        next_theta = expected_components + alpha * prior
        next_theta /= next_theta.sum()
        converged = abs(next_theta - theta).max() < 1e-8
        theta = next_theta
        if converged:
            break
    if not np.isfinite(theta).all() or (theta <= 0).any() or abs(theta.sum() - 1) >= 1e-12:
        raise RuntimeError("Invalid packet-size density solution")
    return theta


def density_scores(model, items, protocol):
    result = np.zeros((len(items), 6))
    available = np.zeros(len(items), int)
    for (transport, direction), group in model["groups"].items():
        ids = [
            i
            for i, item in enumerate(items)
            if protocol[i] == transport and item[direction] is not None and (item[direction].shape[1] > 0)
        ]
        if not ids:
            continue
        rep = representation([items[i][direction] for i in ids])
        matrix = basis(rep["sizes"], group["centers"], model["sigma"], group["geometric_mean"])
        pooled = likelihood(rep, matrix, group["pooled"])[2]
        for j, c in enumerate(CLASSES):
            result[ids, j] += likelihood(rep, matrix, group["classes"][c]["theta"])[2] - pooled
        available[ids] += 1
    for i, t in enumerate(protocol):
        for j, c in enumerate(CLASSES):
            if not model["support"][t, c]:
                result[i, j] = -1000000.0
    assert np.isfinite(result).all()
    return (result, available)


def fit_density(items, labels, protocol, sigma, alpha):
    labels = np.asarray(labels)
    protocol = np.asarray(protocol)
    groups = {}
    for transport in sorted(set(protocol)):
        for direction in ("up", "down"):
            ids = [
                i
                for i, item in enumerate(items)
                if protocol[i] == transport and item[direction] is not None and (item[direction].shape[1] > 0)
            ]
            if not ids:
                continue
            bags = [items[i][direction] for i in ids]
            rep = representation(bags)
            centers = rep["sizes"].copy()
            mean = float(np.mean([a.mean() for a in bags]))
            matrix = basis(rep["sizes"], centers, sigma, mean)
            prior = np.r_[np.repeat(0.9 / len(centers), len(centers)), 0.1]
            pooled = optimize(rep, matrix, prior, 10.0)
            classes = {}
            for app in CLASSES:
                chosen = [i for i in ids if labels[i] == app]
                if chosen:
                    crep = representation([items[i][direction] for i in chosen])
                    cm = basis(crep["sizes"], centers, sigma, mean)
                    theta = optimize(crep, cm, pooled, alpha)
                else:
                    theta = pooled.copy()
                classes[app] = dict(theta=theta)
            groups[transport, direction] = dict(
                centers=centers,
                geometric_mean=mean,
                pooled=pooled,
                classes=classes,
            )
    return dict(
        groups=groups,
        sigma=float(sigma),
        alpha=float(alpha),
        support={
            (t, c): int(((protocol == t) & (labels == c)).sum()) for t in ("tcp", "udp") for c in CLASSES
        },
    )
