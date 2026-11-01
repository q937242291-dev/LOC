"""Combine graded outcomes, repeat statistics and recorded experimental controls."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .analysis import (compare_operating_points, highest_baseline_comparison,
                       paired_comparison, repeat_statistics, resolution_interval,
                       _bootstrap_options, _group_label, _task_id)
from .evaluation import aggregate
from .run_design import audit_run_design


def analysis_report(records: Iterable[Mapping[str, Any]], *, locus_system: str = "LOCUS",
                    expected_repeats: int = 5, samples: int = 2000,
                    confidence: float = .95, seed: int = 0,
                    task_rosters: Mapping[str, list[str]] | None = None) -> dict[str, Any]:
    """Summarize graded attempts, repeat variation and paired comparisons.

    Each input is one method's independently graded attempts. Highest-baseline
    comparisons treat every other supplied system as a baseline candidate;
    component studies also report every named reference-vs-condition pair.
    """
    if type(expected_repeats) is not int or expected_repeats < 1:
        raise ValueError("expected_repeats must be a positive integer")
    if not isinstance(locus_system, str) or not locus_system:
        raise ValueError("locus_system must be a nonempty system name")
    _bootstrap_options(samples, confidence, seed)
    rows = list(records)
    runs = range(expected_repeats)
    options = dict(expected_repeats=runs, samples=samples, confidence=confidence, seed=seed)
    methods: dict[tuple, list] = defaultdict(list)
    operating_groups = defaultdict(list)
    observed_tasks = defaultdict(set)
    for row in rows:
        key = tuple(_group_label(row, name) for name in ("benchmark", "model", "system"))
        if not key[2]:
            raise ValueError("system must be a nonempty string")
        methods[key].append(row)
        operating_groups[key[:2]].append(row)
        observed_tasks[key[:2]].add(_task_id(row.get("instance_id")))
    if task_rosters is not None and any(key[0] not in task_rosters for key in methods):
        raise ValueError("Supply the task roster for every benchmark being evaluated")
    rosters = {key: tuple(task_rosters[key[0]]) if task_rosters is not None else sorted(tasks)
               for key, tasks in observed_tasks.items()}
    groups = aggregate(rows)
    repeats = []
    statistics_by_method = {}
    for (benchmark, model, system), attempts in sorted(methods.items(), key=lambda item: str(item[0])):
        statistics = repeat_statistics(attempts, expected_repeats=runs,
            expected_tasks=rosters[(benchmark, model)])
        statistics_by_method[(benchmark, model, system)] = statistics
        interval = resolution_interval(attempts, samples=samples, confidence=confidence, seed=seed) \
            if statistics["complete"] else None
        repeats.append(dict(benchmark=benchmark, model=model, system=system,
                            **statistics, resolution_interval=interval))
    paired, highest = [], []
    for key, attempts in sorted(operating_groups.items(), key=lambda item: str(item[0])):
        selected = dict(options, expected_tasks=rosters[key])
        for system in sorted({r.get("system") for r in attempts if r.get("system") != locus_system}, key=str):
            for comparison in paired_comparison(attempts, locus_system, system, **selected):
                comparison["design_audit"] = audit_run_design(
                    methods.get((*key, locus_system), []), methods.get((*key, system), []))
                paired.append(comparison)
        highest.extend(highest_baseline_comparison(attempts, locus_system, **selected))
    comparable_groups = []
    for group in groups:
        key = tuple(group[name] for name in ("benchmark", "model", "system"))
        comparable = dict(group)
        if not statistics_by_method[key]["balanced_roster"]:
            for metric in ("resolution_pct", "normalized_cost_mean", "tokens_mean"):
                comparable[metric] = None
        comparable_groups.append(comparable)
    comparisons = compare_operating_points(comparable_groups, locus_name=locus_system)
    for comparison in comparisons:
        key = comparison["benchmark"], comparison["model"]
        comparison["common_roster_complete"] = all(
            statistics["balanced_roster"] for method, statistics in statistics_by_method.items()
            if method[:2] == key)
    return dict(groups=groups, comparisons=comparisons,
                repeats=repeats, paired_comparisons=paired,
                highest_baseline_comparisons=highest,
                statistical_protocol=dict(expected_repeats=expected_repeats, confidence=confidence,
                    bootstrap_samples=samples, bootstrap_seed=seed, cluster="instance_id",
                    repeat_variance_ddof=1, resolution_unit="percentage points",
                    roster_source="explicit" if task_rosters is not None else "observed_input_union"))
