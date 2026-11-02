"""Paired statistics. Missing, unpaired or nonfinite samples are errors."""
import math
import random
import statistics


def percentile(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError("Nonempty values and quantile in [0,1] required")
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_summary(pairs, *, bootstrap=2000, seed=7):
    if type(bootstrap) is not int or bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    pairs = list(pairs)
    if len(pairs) < 2:
        raise ValueError("At least two independent paired samples required")
    for pair in pairs:
        if len(pair) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in pair):
            raise ValueError("Each pair must contain positive finite baseline/accelerated times")
    saved = [b - a for b, a in pairs]
    randomizer = random.Random(seed)
    means, medians = [], []
    for _ in range(bootstrap):
        draw = randomizer.choices(saved, k=len(saved))
        means.append(statistics.mean(draw))
        medians.append(statistics.median(draw))
    return {"paired_samples": len(pairs), "baseline_median_ms": statistics.median(b for b, _ in pairs),
            "accelerated_median_ms": statistics.median(a for _, a in pairs),
            "median_paired_speedup": statistics.median(b / a for b, a in pairs),
            "mean_saved_ms": statistics.mean(saved), "median_saved_ms": statistics.median(saved),
            "paired_mean_saved_ms_ci95": [percentile(means, .025), percentile(means, .975)],
            "paired_median_saved_ms_ci95": [percentile(medians, .025), percentile(medians, .975)],
            "raw_paired_saved_ms": saved, "bootstrap_draws": bootstrap, "bootstrap_seed": seed}


def join_pairs(records):
    """Do not pair by row position: require explicit shared experiment identities."""
    grouped = {}
    for record in records:
        if record.get("error"):
            raise ValueError("Errored records must be retried before statistical admission")
        if record.get("mode") not in ("baseline", "accelerated"):
            continue
        fields = ("pair_id", "task", "trial", "seed", "initial_state_hash", "input_hash",
                  "checkpoint_id", "protocol_hash", "device_id")
        if any(record.get(key) is None for key in fields):
            raise ValueError("Missing pairing identity")
        identity = tuple(record[key] for key in fields)
        modes = grouped.setdefault(identity, {})
        if record["mode"] in modes:
            raise ValueError("Duplicate paired record")
        modes[record["mode"]] = record["wall_ms"]
    if not grouped or any(set(modes) != {"baseline", "accelerated"} for modes in grouped.values()):
        raise ValueError("Unpaired or mismatched records")
    return [(modes["baseline"], modes["accelerated"]) for modes in grouped.values()]
