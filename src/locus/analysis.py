"""Task-cluster uncertainty, budget-vote diagnostics and operating comparisons."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from math import isfinite, sqrt
import random
from statistics import fmean, stdev, variance
from typing import Any

from .evaluation import summarize


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * quantile
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (pos - low) * (ordered[high] - ordered[low])


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Pearson inputs must have equal lengths")
    if len(xs) < 2:
        return None
    xm, ym = fmean(xs), fmean(ys)
    xx = sum((x - xm) ** 2 for x in xs)
    yy = sum((y - ym) ** 2 for y in ys)
    return None if xx == 0 or yy == 0 else sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sqrt(xx * yy)


def task_cluster_bootstrap(
    records: Iterable[Mapping[str, Any]], statistic: Callable[[list[Mapping[str, Any]]], float | None],
    *, samples: int = 2000, confidence: float = 0.95, seed: int = 0,
) -> dict[str, Any]:
    """Resample benchmark/task clusters, retaining every repeat and method."""
    rows = list(records)
    _bootstrap_options(samples, confidence, seed)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(_group_label(row, "benchmark"), _task_id(row.get("instance_id")))].append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    estimates = []
    def checked(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Bootstrap statistics must be finite numbers or null")
        return float(value)
    estimate = checked(statistic(rows)) if rows else None
    for _ in range(samples if keys and estimate is not None else 0):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in groups[key]]
        value = checked(statistic(sample))
        if value is not None:
            estimates.append(float(value))
    tail = (1 - confidence) / 2
    return {"estimate": estimate,
            "lower": percentile(estimates, tail), "upper": percentile(estimates, 1 - tail),
            "confidence": confidence, "samples": samples, "valid_samples": len(estimates),
            "task_clusters": len(keys), "seed": seed}


def resolution_interval(records: Iterable[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Resolution uncertainty for one benchmark/model/system attempt group."""
    rows = list(records)
    labels = {tuple(_group_label(row, key) for key in ("benchmark", "model", "system"))
              for row in rows}
    if len(labels) > 1:
        raise ValueError("resolution_interval requires one benchmark/model/system group")
    _attempt_index(rows)
    return task_cluster_bootstrap(rows, lambda sample: summarize(sample)["resolution_pct"], **kwargs)


def compare_operating_points(summaries: Iterable[Mapping[str, Any]], locus_name: str = "LOCUS",
                             *, quality_reference: str | None = None,
                             epsilon: float = .01) -> list[dict[str, Any]]:
    """Table 2 comparisons and Eq. 3 quality checks on evaluated group means.

    Resolution ties prefer lower mean cost. A baseline's missing telemetry does
    not remove it from quality comparisons. ``epsilon`` is a resolution fraction,
    so .01 permits one percentage point of loss. ``quality_reference`` chooses
    the fixed reference system; by default the highest-resolution baseline is
    used. System names match exactly. Identifying the cheapest baseline requires
    every supplied baseline's mean cost. This diagnoses evaluated group means.
    """
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or
            not isfinite(epsilon) or not 0 <= epsilon <= 1):
        raise ValueError("epsilon must be a resolution fraction in [0, 1]")
    if not isinstance(locus_name, str) or not locus_name:
        raise ValueError("locus_name must be a nonempty system name")
    if quality_reference is not None and (not isinstance(quality_reference, str) or not quality_reference):
        raise ValueError("quality_reference must be a nonempty system name")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in summaries:
        key = tuple(_group_label(row, name) for name in ("benchmark", "model", "system"))
        if not key[2]:
            raise ValueError("system must be a nonempty string")
        if key in seen:
            raise ValueError(f"Duplicate summary group: {key!r}")
        seen.add(key)
        for field in ("resolution_pct", "normalized_cost_mean", "tokens_mean"):
            value = row.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                                      not isfinite(value) or value < 0 or
                                      (field == "resolution_pct" and value > 100)):
                raise ValueError(f"{field} must be a finite nonnegative number or null")
        groups[key[:2]].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        locus = next((r for r in rows if r["system"] == locus_name), None)
        base = [r for r in rows if r is not locus]
        if locus is None or not base:
            continue
        graded = [r for r in base if r.get("resolution_pct") is not None]
        priced = [r for r in base if r.get("normalized_cost_mean") is not None]
        complete_quality = len(graded) == len(base)
        highest = max((r["resolution_pct"] for r in graded), default=None) if complete_quality else None
        tied = [r for r in graded if r["resolution_pct"] == highest]
        best = None
        if len(tied) == 1:
            best = tied[0]
        elif tied and all(r.get("normalized_cost_mean") is not None for r in tied):
            best = min(tied, key=lambda r: (r["normalized_cost_mean"], str(r.get("system", ""))))
        complete_cost = len(priced) == len(base)
        cheapest = min(priced, key=lambda r: (r["normalized_cost_mean"], r["system"])) if complete_cost else None
        def saving(metric: str, comparator: Mapping[str, Any] | None) -> float | None:
            value, reference = locus.get(metric), None if comparator is None else comparator.get(metric)
            return None if value is None or reference in (None, 0) else 100 * (1 - value / reference)
        reference_row = next((r for r in base if r["system"] == quality_reference), None) if quality_reference else best
        reference_resolution = reference_row.get("resolution_pct") if reference_row else (highest if quality_reference is None else None)
        delta = None if locus.get("resolution_pct") is None or highest is None else locus["resolution_pct"] - highest
        quality_delta = None if locus.get("resolution_pct") is None or reference_resolution is None else locus["resolution_pct"] - reference_resolution
        results.append({"benchmark": key[0], "model": key[1],
                        "highest_resolution_baseline": None if best is None else best.get("system"),
                        "highest_resolution_candidates": [r.get("system") for r in tied],
                        "resolution_delta_pp": delta,
                        "tokens_saved_pct": saving("tokens_mean", best),
                        "cost_saved_pct": saving("normalized_cost_mean", best),
                        "lowest_cost_baseline": None if cheapest is None else cheapest.get("system"),
                        "baseline_costs_complete": complete_cost,
                        "cost_saved_vs_cheapest_pct": saving("normalized_cost_mean", cheapest),
                        "quality_reference": quality_reference or (best.get("system") if best else "highest-resolution baseline"),
                        "quality_delta_pp": quality_delta, "quality_tolerance_pp": 100 * epsilon,
                        "quality_floor_met": None if quality_delta is None else quality_delta >= -100 * epsilon - 1e-12})
    return results


def _repeat_id(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("repeat must be a nonnegative integer, not a boolean")
    return value


def _task_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("instance_id must be a nonempty string")
    return value


def _grade(row: Mapping[str, Any]) -> bool | None:
    grade = row.get("resolved")
    if grade is not True and grade is not False and grade is not None:
        raise ValueError("resolved must be true, false or null")
    return grade


def _expected(values, validate, label):
    if values is None:
        return None
    result = tuple(validate(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be nonempty and contain no duplicates")
    return frozenset(result)


def _group_label(row, name):
    value = row.get(name, "")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _attempt_index(rows):
    indexed = {}
    for row in rows:
        key = _task_id(row.get("instance_id")), _repeat_id(row.get("repeat"))
        if key in indexed:
            raise ValueError(f"Duplicate attempt for instance_id={key[0]!r}, repeat={key[1]}")
        _grade(row)
        indexed[key] = row
    return indexed


def _roster(indexes, expected_repeats, expected_tasks):
    observed = set().union(*(set(index) for index in indexes))
    observed_repeats = {repeat for _, repeat in observed}
    observed_tasks = {task for task, _ in observed}
    if expected_repeats is not None and not observed_repeats <= expected_repeats:
        raise ValueError("Observed repeat is outside expected_repeats")
    if expected_tasks is not None and not observed_tasks <= expected_tasks:
        raise ValueError("Observed task is outside expected_tasks")
    repeats = sorted(observed_repeats if expected_repeats is None else expected_repeats)
    tasks = sorted(observed_tasks if expected_tasks is None else expected_tasks)
    return repeats, tasks, {(task, repeat) for task in tasks for repeat in repeats}


def _grade_counts(rows):
    rows = list(rows)
    graded = sum(_grade(row) is not None for row in rows)
    successes = sum(_grade(row) is True for row in rows)
    return {"attempts": len(rows), "graded_attempts": graded,
            "ungraded_attempts": len(rows) - graded, "known_successes": successes,
            "successes": successes if graded == len(rows) else None,
            "resolution_pct": 100 * successes / len(rows) if rows and graded == len(rows) else None}


def repeat_statistics(
    records: Iterable[Mapping[str, Any]], metric: str = "resolution_pct", *,
    expected_repeats: Iterable[int] | None = None,
    expected_tasks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Per-repeat counts, metric means and sample variance (denominator r-1).

    Supply one benchmark/model/system group. Attempts must have unique exact
    (instance_id, repeat) keys, integer repeat IDs and boolean/null grades.
    ``expected_repeats=range(5)`` checks the five-repeat protocol; otherwise
    repeat IDs are inferred from the input. ``expected_tasks`` can specify the
    full benchmark roster. The complete graded roster supplies mean/SD/variance;
    counts_by_repeat reports the number of submitted and graded attempts.
    """
    rows = list(records)
    labels = {tuple(_group_label(row, key) for key in ("benchmark", "model", "system"))
              for row in rows}
    if len(labels) > 1:
        raise ValueError("repeat_statistics requires one benchmark/model/system group")
    indexed = _attempt_index(rows)
    repeats, tasks, slots = _roster(
        (indexed,), _expected(expected_repeats, _repeat_id, "expected_repeats"),
        _expected(expected_tasks, _task_id, "expected_tasks"))
    by_repeat = {}
    counts = {}
    for repeat in repeats:
        current = [row for (task, run), row in indexed.items() if run == repeat]
        present_tasks = {task for task, run in indexed if run == repeat}
        roster_complete = present_tasks == set(tasks) and bool(tasks)
        value = summarize(current).get(metric) if current else None
        value = value if (not isinstance(value, bool) and isinstance(value, (int, float))
                          and isfinite(value) and roster_complete) else None
        by_repeat[str(repeat)] = value
        counts[str(repeat)] = {**_grade_counts(current), "expected_attempts": len(tasks),
                               "missing_attempts": len(tasks) - len(current),
                               "roster_complete": roster_complete}
    known = [value for value in by_repeat.values() if value is not None]
    balanced = bool(slots) and set(indexed) == slots
    metric_complete = bool(repeats) and len(known) == len(repeats)
    complete = balanced and metric_complete
    return {"metric": metric, "by_repeat": by_repeat, "counts_by_repeat": counts,
            "mean": fmean(known) if complete else None,
            "sample_std": stdev(known) if len(known) > 1 and complete else None,
            "sample_variance": variance(known) if len(known) > 1 and complete else None,
            "repeat_count": len({repeat for _, repeat in indexed}),
            "expected_repeat_count": len(repeats), "known_repeat_count": len(known),
            "task_count": len(tasks), "attempts": len(rows), "expected_attempts": len(slots),
            "missing_attempts": len(slots) - len(rows), "balanced_roster": balanced,
            "grading_complete": bool(rows) and all(_grade(row) is not None for row in rows),
            "metric_complete": metric_complete, "complete": complete,
            "five_repeat_complete": complete and len(repeats) == 5}


def _paired_counts(left, right, slots):
    matched = set(left) & set(right)
    known = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for key in matched:
        a, b = _grade(left[key]), _grade(right[key])
        if a is None or b is None:
            continue
        cell = "both" if a and b else "left_only" if a else "right_only" if b else "neither"
        known[cell] += 1
    balanced = bool(slots) and set(left) == set(right) == slots
    complete = balanced and sum(known.values()) == len(slots)
    left_counts, right_counts = _grade_counts(left.values()), _grade_counts(right.values())
    delta = 100 * (known["left_only"] - known["right_only"]) / len(slots) if complete else None
    return {"left": left_counts, "right": right_counts, "expected_pairs": len(slots),
            "matched_attempts": len(matched), "graded_pairs": sum(known.values()),
            "missing_left": len(slots - set(left)), "missing_right": len(slots - set(right)),
            "unpaired_left": len(set(left) - set(right)), "unpaired_right": len(set(right) - set(left)),
            **{cell: value if complete else None for cell, value in known.items()},
            "known_pair_counts": known, "balanced_roster": balanced, "complete": complete,
            "grading_complete": bool(left or right) and
                all(_grade(row) is not None for row in (*left.values(), *right.values())),
            "resolution_delta_pp": delta}


def _bootstrap_options(samples, confidence, seed):
    if (type(samples) is not int or samples < 1 or
            isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
            not isfinite(confidence) or not 0 < confidence < 1 or type(seed) is not int):
        raise ValueError("samples must be a positive integer, confidence in (0,1), and seed an integer")


def _delta_interval(estimate, estimates, *, samples, confidence, seed, tasks, **metadata):
    tail = (1 - confidence) / 2
    return {"estimate": estimate, "lower": percentile(estimates, tail),
            "upper": percentile(estimates, 1 - tail), "confidence": confidence,
            "samples": samples, "valid_samples": len(estimates), "task_clusters": len(tasks),
            "seed": seed, **metadata}


def paired_comparison(
    records: Iterable[Mapping[str, Any]], left_system: str, right_system: str, *,
    expected_repeats: Iterable[int] | None = None,
    expected_tasks: Iterable[str] | None = None,
    samples: int = 2000, confidence: float = .95, seed: int = 0,
) -> list[dict[str, Any]]:
    """Fixed named comparison, per-repeat discordance and paired cluster CI.

    Results are grouped by benchmark/model. System names match exactly; the
    comparator is supplied explicitly and remains fixed in every bootstrap.
    A pair joins the exact (instance_id, repeat) key, with one boolean/null
    grade per method. Complete graded pairs supply the delta and CI.
    ``left_only``/``right_only`` count discordant successes; missing_left and
    missing_right count absent attempt records. S/N counts accompany each run.

    The rectangular roster uses all observed tasks/repeats, or the explicitly
    supplied expected sets. Pass ``expected_repeats=range(5)`` for five repeats.
    To check several benchmarks with different expected task rosters, call
    separately per benchmark. Empty input returns an empty result list.

    Delta is left resolution minus right resolution, in percentage points.
    Resampling draws tasks with replacement and retains every repeat and both
    methods for each drawn task. Intervals are percentile intervals of the pooled delta,
    preserving the paired attempt denominator in every resample. Per-repeat
    intervals use the same task multiplicities, restricted to that repeat.
    These intervals are conditional on the explicitly named comparator.
    """
    if (not isinstance(left_system, str) or not left_system or
            not isinstance(right_system, str) or not right_system or left_system == right_system):
        raise ValueError("Specify two different nonempty system names")
    _bootstrap_options(samples, confidence, seed)
    expected_runs = _expected(expected_repeats, _repeat_id, "expected_repeats")
    expected_ids = _expected(expected_tasks, _task_id, "expected_tasks")
    groups = defaultdict(lambda: {left_system: [], right_system: []})
    for row in records:
        system = row.get("system")
        if system not in (left_system, right_system):
            continue
        key = _group_label(row, "benchmark"), _group_label(row, "model")
        groups[key][system].append(row)
    results = []
    for (benchmark, model), methods in sorted(groups.items()):
        left, right = _attempt_index(methods[left_system]), _attempt_index(methods[right_system])
        repeats, tasks, slots = _roster((left, right), expected_runs, expected_ids)
        counts = _paired_counts(left, right, slots)
        per_repeat = {}
        for repeat in repeats:
            per_repeat[str(repeat)] = _paired_counts(
                {key: row for key, row in left.items() if key[1] == repeat},
                {key: row for key, row in right.items() if key[1] == repeat},
                {(task, repeat) for task in tasks})
        estimates, repeat_estimates = [], {repeat: [] for repeat in repeats}
        run_deltas = {
            repeat: {task: int(_grade(left[(task, repeat)])) - int(_grade(right[(task, repeat)]))
                     for task in tasks}
            for repeat in repeats if per_repeat[str(repeat)]["complete"]}
        if run_deltas:
            pooled_deltas = {task: sum(values[task] for values in run_deltas.values())
                             for task in tasks} if counts["complete"] else {}
            rng = random.Random(seed)
            for _ in range(samples):
                weights = Counter(rng.choices(tasks, k=len(tasks)))
                if counts["complete"]:
                    estimates.append(100 * sum(weight * pooled_deltas[task] for task, weight in weights.items()) / len(slots))
                for repeat, values in run_deltas.items():
                    repeat_estimates[repeat].append(
                        100 * sum(weight * values[task] for task, weight in weights.items()) / len(tasks))
        interval_options = dict(samples=samples, confidence=confidence, seed=seed, tasks=tasks,
                                conditional_on_comparator=True, comparator_system=right_system)
        interval = _delta_interval(counts["resolution_delta_pp"], estimates, **interval_options)
        for repeat in repeats:
            run = per_repeat[str(repeat)]
            run["delta_interval"] = _delta_interval(
                run["resolution_delta_pp"], repeat_estimates[repeat], **interval_options)
        results.append({"benchmark": benchmark, "model": model,
                        "left_system": left_system, "right_system": right_system,
                        "comparator_system": right_system, "comparison": "fixed_named",
                        **counts, "by_repeat": per_repeat, "delta_interval": interval,
                        "task_count": len(tasks), "repeat_count": len(repeats),
                        "five_repeat_complete": counts["complete"] and len(repeats) == 5})
    return results


def highest_baseline_comparison(
    records: Iterable[Mapping[str, Any]], locus_system: str = "LOCUS", *,
    expected_repeats: Iterable[int] | None = None,
    expected_tasks: Iterable[str] | None = None,
    samples: int = 2000, confidence: float = .95, seed: int = 0,
) -> list[dict[str, Any]]:
    """Compare to highest pooled baseline with selection-aware pooled delta CI.

    All other supplied systems are comparator candidates. Each benchmark/model
    group must have the same complete graded task/repeat roster for every
    candidate and LOCUS. The point comparator maximizes pooled resolution;
    quality ties use lower complete mean cost, then system name. A unique
    quality maximum identifies the comparator directly. Pending costs retain
    the tied candidate set together with the pooled delta/CI.

    Every pooled bootstrap replicate uses shared task multiplicities for all
    systems and reselects the maximum baseline resolution. Thus the pooled
    interval covers the LOCUS-minus-maximum estimator. Per-repeat counts and
    per-repeat intervals, when a point comparator is identified, keep that
    observed pooled comparator fixed and are explicitly conditional on it.
    They are a different estimand from maximizing a baseline within each run.
    """
    if not isinstance(locus_system, str) or not locus_system:
        raise ValueError("locus_system must be a nonempty system name")
    _bootstrap_options(samples, confidence, seed)
    expected_runs = _expected(expected_repeats, _repeat_id, "expected_repeats")
    expected_ids = _expected(expected_tasks, _task_id, "expected_tasks")
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        system = row.get("system")
        if not isinstance(system, str) or not system:
            raise ValueError("system must be a nonempty string")
        key = _group_label(row, "benchmark"), _group_label(row, "model")
        groups[key][system].append(row)
    results = []
    for (benchmark, model), methods in sorted(groups.items()):
        indexes = {system: _attempt_index(rows) for system, rows in methods.items()}
        indexes.setdefault(locus_system, {})
        baselines = sorted(system for system in indexes if system != locus_system)
        repeats, tasks, slots = _roster(tuple(indexes.values()), expected_runs, expected_ids)
        complete = bool(slots) and bool(baselines) and all(
            set(index) == slots and all(_grade(row) is not None for row in index.values())
            for index in indexes.values())
        candidates, selected, delta = [], None, None
        method_counts = {system: _grade_counts(index.values()) for system, index in sorted(indexes.items())}
        estimates = []
        if complete:
            successes = {system: counts["successes"] for system, counts in method_counts.items()}
            maximum = max(successes[system] for system in baselines)
            candidates = [system for system in baselines if successes[system] == maximum]
            delta = 100 * (successes[locus_system] - maximum) / len(slots)
            if len(candidates) == 1:
                selected = candidates[0]
            else:
                costs = {system: summarize(methods[system])["normalized_cost_mean"] for system in candidates}
                if all(cost is not None for cost in costs.values()):
                    selected = min(candidates, key=lambda system: (costs[system], system))
            cluster_successes = {
                system: {task: sum(_grade(index[(task, repeat)]) is True for repeat in repeats)
                         for task in tasks} for system, index in indexes.items()}
            rng = random.Random(seed)
            for _ in range(samples):
                weights = Counter(rng.choices(tasks, k=len(tasks)))
                successes_sample = {
                    system: sum(weight * values[task] for task, weight in weights.items())
                    for system, values in cluster_successes.items()}
                estimates.append(100 * (successes_sample[locus_system] -
                    max(successes_sample[system] for system in baselines)) / len(slots))
        paired = {}
        if selected is not None:
            paired = paired_comparison(
                methods[locus_system] + methods[selected], locus_system, selected,
                expected_repeats=repeats, expected_tasks=tasks,
                samples=samples, confidence=confidence, seed=seed)[0]
        run_counts = {
            str(repeat): {system: _grade_counts(row for (task, run), row in index.items() if run == repeat)
                          for system, index in sorted(indexes.items())} for repeat in repeats}
        interval = _delta_interval(
            delta, estimates, samples=samples, confidence=confidence, seed=seed, tasks=tasks,
            conditional_on_comparator=False, comparator_reselected=True,
            comparator_candidates=baselines)
        results.append({**paired, "benchmark": benchmark, "model": model,
                        "left_system": locus_system, "right_system": selected,
                        "comparator_system": selected,
                        "comparison": "highest_baseline_reselected",
                        "highest_resolution_baseline": selected,
                        "highest_resolution_candidates": candidates,
                        "baseline_candidates": baselines, "method_counts": method_counts,
                        "counts_by_repeat": run_counts,
                        "left": method_counts[locus_system],
                        "right": method_counts[selected] if selected is not None else None,
                        "comparator_identified": selected is not None,
                        "complete": complete, "resolution_delta_pp": delta,
                        "delta_interval": interval, "by_repeat": paired.get("by_repeat", {}),
                        "task_count": len(tasks), "repeat_count": len(repeats),
                        "expected_pairs": len(slots),
                        "five_repeat_complete": complete and len(repeats) == 5})
    return results


def make_agreement_cohorts(tasks: Iterable[Mapping[str, Any]], *, cohorts: int = 45,
                           per_class: int = 10, seed: int = 0) -> list[list[str]]:
    """Ten distinct tasks per L/M/H class in each of 45 cohorts.

    Each task mapping has instance_id, budget_class, agreement and resolved
    (or a resolution fraction). Tasks may recur across different cohorts.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for row in tasks:
        key = str(row["instance_id"])
        if key in seen:
            raise ValueError("Cohort input must have one row per task")
        seen.add(key)
        groups[str(row["budget_class"])].append(key)
    if cohorts < 1 or per_class < 1 or any(len(groups[c]) < per_class for c in ("L", "M", "H")):
        raise ValueError("Each budget class must contain at least per_class tasks")
    rng = random.Random(seed)
    return [[key for cls in ("L", "M", "H") for key in rng.sample(sorted(groups[cls]), per_class)]
            for _ in range(cohorts)]


def agreement_correlation(tasks: Iterable[Mapping[str, Any]], cohorts: Sequence[Sequence[str]],
                          *, samples: int = 2000, seed: int = 0,
                          confidence: float = 0.95) -> dict[str, Any]:
    """Cohort Pearson r and task-cluster CI with shared recurrence weights.

    Every sampled task receives one multiplicity reused in every cohort that
    contains it; recurring memberships are not independent observations.
    """
    rows = list(tasks)
    by_id = {str(r["instance_id"]): r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Expected one aggregate row per task")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    ids = sorted({str(key) for cohort in cohorts for key in cohort})
    if any(key not in by_id for key in ids):
        raise ValueError("Cohort contains an unknown task")
    def correlation(weights: Mapping[str, int]) -> float | None:
        agreements, rates = [], []
        for cohort in cohorts:
            denominator = sum(weights.get(str(key), 0) for key in cohort)
            if denominator == 0:
                continue
            a = y = 0.0
            for item in cohort:
                key = str(item)
                row, weight = by_id[key], weights.get(key, 0)
                outcome = row.get("resolution", row.get("resolved"))
                if outcome is None:
                    raise ValueError("Cohort correlation requires graded tasks")
                a += weight * float(row["agreement"])
                y += weight * float(outcome)
            agreements.append(a / denominator)
            rates.append(y / denominator)
        return pearson(agreements, rates)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples if ids else 0):
        value = correlation(Counter(rng.choices(ids, k=len(ids))))
        if value is not None:
            estimates.append(value)
    tail = (1 - confidence) / 2
    return {"pearson_r": correlation(dict.fromkeys(ids, 1)),
            "lower": percentile(estimates, tail), "upper": percentile(estimates, 1 - tail),
            "confidence": confidence, "samples": samples, "valid_samples": len(estimates),
            "task_clusters": len(ids), "cohort_count": len(cohorts), "seed": seed}


def allocation_summary(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Task-level vote concentration, class counts and higher-class tie promotion."""
    rows = list(tasks)
    if len({str(row["instance_id"]) for row in rows}) != len(rows):
        raise ValueError("Allocation counts require one row per unique task")
    counts = Counter(str(x["budget_class"]) for x in rows)
    agreements = [float(x["agreement"]) for x in rows]
    ties, promoted_high = 0, 0
    for row in rows:
        votes = row.get("votes", {})
        labels = votes.values() if isinstance(votes, Mapping) else votes
        values = Counter(label for label in labels if label in {"L", "M", "H"})
        if values:
            top = max(values.values())
            tied = [k for k, v in values.items() if v == top]
            ties += len(tied) > 1
            promoted_high += len(tied) > 1 and "H" in tied and row["budget_class"] == "H"
    n = len(rows)
    return {"tasks": n, "class_counts": {c: counts[c] for c in ("L", "M", "H")},
            "agreement_q1": percentile(agreements, .25), "agreement_median": percentile(agreements, .5),
            "agreement_q3": percentile(agreements, .75), "ties": ties,
            "tie_pct": 100 * ties / n if n else None, "promoted_high": promoted_high,
            "promoted_high_share_of_high_pct": 100 * promoted_high / counts["H"] if counts["H"] else None,
            "promoted_high_share_of_all_pct": 100 * promoted_high / n if n else None}
