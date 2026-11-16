"""Packet-flow classification with fixed public task and class order."""

CLASSES = ("facebook", "google", "instagram", "mozilla", "x", "youtube")
TASKS = ("task1_static", "task2_realtime_prefix5")


def fold_indices(raw, entry):
    """Resolve a validated fold plan without relying on dataframe row indexes."""
    lookup = {flow: i for i, flow in enumerate(raw.flow_id)}
    fit = [lookup[f] for f in entry["fit_ids"]]
    held = [lookup[f] for f in entry["held_ids"]]
    if set(fit) & set(held) or set(fit) | set(held) != set(range(len(raw))):
        raise ValueError("Fold plan must partition all training rows")
    return (fit, held)
