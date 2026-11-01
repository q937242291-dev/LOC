# Evaluation and Experimental Analysis

## Experimental Settings

The paper evaluates SWE-bench Lite (300 tasks), Pro V1 (731 tasks), and Pro V2 (642 tasks). Five runs per task yield 1,500, 3,655, and 3,210 attempts, respectively. Results are aggregated separately for each release.

The paper specifies common sampling requests of `temperature=0.7` and `top_p=1.0`. Each run uses a 128,000-token input limit, a 32,768-token output limit, and a limit of 250 model/tool calls. The default evidence-packet cap is 16,000 tokens, with five candidate edit locations. The L/M/H exploration profiles allow 64/128/210 calls and 1/2/3 relation-expansion layers. Task deadlines are 7,200 seconds for Pro V1 and 3,000 seconds for Pro V2. Workers use 8 vCPUs and 32 GiB of memory.

The DeepSeek configuration disables thinking so that the temperature parameter takes effect. The GPT-5.2 configuration uses `reasoning_effort="none"`. Native Sonnet 4.5 requests send only `temperature=0.7`; the omitted `top_p` uses the Messages API default of 0.99. `model_runtime.requests` records request fields, requested seeds, returned model names, and backend fingerprints. See [Sampling Settings](upstream_sampling.md) for the interfaces and upstream defaults.

## Attempt Records and Aggregation

`locus.evaluation.aggregate` accepts an iterable of attempt records and groups them by `benchmark`, `model`, and `system`. Each record represents one run of one task. The main fields are:

| Field | Contents |
| --- | --- |
| `instance_id` | Task identifier |
| `repeat` | Repeat index |
| `benchmark`, `model`, `system` | Benchmark, model, and method identifiers |
| `resolved` | Boolean outcome from official grading tests |
| `file_predictions` | Ordered predicted file paths |
| `ledger` | Complete-task resource accounting returned by `Ledger.summary()` |
| `evidence_metrics` | Evidence statistics returned by `evidence_metrics(...)` |

```python
import json
from pathlib import Path
from locus.evaluation import aggregate
from locus.analysis import compare_operating_points

records = [json.loads(line) for line in Path("results/attempts.jsonl")
           .read_text(encoding="utf-8").splitlines() if line.strip()]
groups = aggregate(records)
print(json.dumps(groups, ensure_ascii=False, indent=2))
print(json.dumps(compare_operating_points(groups), ensure_ascii=False, indent=2))
```

`aggregate` returns a list of group summaries. `summarize(records)` directly summarizes one supplied group of attempts. Aggregation pools attempt counts, successes, and resource consumption before calculating the corresponding ratios. All five runs of each task enter the denominator.

Table 2 compares quality against the baseline with the highest resolution rate; ties select the lower mean cost. The lowest-cost baseline is selected separately. Quality comparisons use official grades, and cost comparisons use complete-task consumption.

The quality tolerance `epsilon=0.01` in Eq. (3) corresponds to one percentage point. For the allocation experiment, explicitly choose broad context as the fixed quality reference:

```python
comparison = compare_operating_points(
    groups, quality_reference="mini-swe-agent", epsilon=0.01,
)
```

`quality_delta_pp` is the evaluated resolution rate of the current method minus that reference. `quality_floor_met` indicates whether the aggregate operating point reaches the reference rate minus one percentage point. `quality_reference` uses the method name in the records; pass `"mini-swe-agent"` for an external mini-swe-agent baseline. This check applies to aggregate results over the evaluated task set. The default quality reference is the baseline with the highest resolution rate, and `resolution_delta_pp` always uses that highest-resolution comparison.

`resolved=null` means grading is pending. `graded_attempts`, `ungraded_attempts`, and `graded_resolution_pct` show grading progress. The complete group's `resolution_pct` becomes available after all attempts are graded. `usage_complete` indicates that provider usage accounting is complete. `known_resources_total` and `known_normalized_cost_total` contain the accounted portions; full cost and token metrics become available when accounting is complete.

## Repair and Resource Metrics

Let `N` be the number of attempts, `S` the number of successes, and `C_i` the normalized complete-task cost of attempt `i`.

| Metric | Definition | Output field |
| --- | --- | --- |
| Resolution | `100 * S / N` | `resolution_pct` |
| Mean cost | `sum(C_i) / N` | `normalized_cost_mean` |
| Cost per success | `sum(C_i) / S` | `cost_per_success` |
| Mean tokens | Sum of uncached input, cached input, and output tokens across all roles, divided by `N` | `tokens_mean` |
| Tokens per success | Complete-task tokens across all attempts, divided by `S` | `tokens_per_success` |
| Million tokens per success | Complete-task tokens across all attempts, divided by `S * 1e6` | `tokens_million_per_success` |
| Mean task time | Total seconds from task start to patch submission, divided by `N` | `elapsed_seconds_mean` |
| Mean calls | Total model and tool calls divided by `N` | `calls_mean` |

Resource consumption from failed attempts contributes to both mean cost and cost per success. Cost per success is `null` when there are no successful attempts.

Tokens per success pools uncached input, cached input, and output tokens across every attempt, including failures and repeated inputs across roles. Divide this pooled total by the number of successful attempts. Both token-per-success fields are `null` when there are no successes, grading is pending, or token accounting is incomplete.

Eq. (5) uses normalized prices of USD 1.25 per million uncached input tokens, USD 0.10 per million cached input tokens, USD 6.00 per million output tokens, and USD 0.06 per worker-hour.

```python
from locus.evaluation import normalized_cost

cost = normalized_cost(
    uncached_input=ledger["uncached_input"],
    cached_input=ledger["cached_input"],
    output=ledger["output"],
    worker_hours=ledger["worker_hours"],
)
```

`worker_hours` accumulates occupancy across workers, charging overlapping workers separately. Task wall time counts overlapping execution once.

The `uncached_input`, `cached_input`, and `output` entries of `token_shares_pct` use all tokens as their denominator. `cached_share_of_input` uses only the sum of uncached and cached input tokens. `request_cache_hit_rate_pct` is calculated from per-request `cache_hit` booleans and is distinct from the cached-input token share.

Step timing accepts explicit `wall_seconds`, `llm_seconds`, and `tool_seconds`, as well as the project's `Ledger` format with `kind` and `seconds`. Each `step_*_mean` is the arithmetic mean over all steps. `step_residual_seconds_mean` is mean Wall minus mean LLM minus mean Tool.

`step_count` is the complete step count, and `known_step_count` is the number of recorded steps. Step timing and request-level cache-hit rates use their complete records as denominators.

### Exploration Diagnostics

Table 5(a) reports exploration per attempt. Exploration comprises the `setup` and `acquisition` phases before patching begins, including indexing, debate, evidence acquisition, and failed actions. Patching, validation, and recovery reads belong to `repair` and are excluded from these exploration totals. Its Calls column uses `exploration_calls_mean`, counting both model and tool actions, including retries. LLM time uses `exploration_llm_seconds_mean` and excludes tools. Model-only and tool-only call counts are supplementary breakdowns:

| Metric | Attempt-level definition | Aggregate mean field |
| --- | --- | --- |
| Exploration actions | All model and tool actions in setup and acquisition | `exploration_calls_mean` |
| Exploration model calls | Model calls in setup and acquisition | `exploration_model_calls_mean` |
| Exploration tool calls | Tool actions in setup and acquisition | `exploration_tool_calls_mean` |
| Exploration LLM time | Model-call seconds in setup and acquisition | `exploration_llm_seconds_mean` |
| Exploration tool time | Tool-action seconds in setup and acquisition | `exploration_tool_seconds_mean` |

Each mean divides the pooled exploration total by all attempts, including failures. The corresponding totals appear in `resources_total` under the same names without `_mean`. `calls_mean`, `llm_seconds_mean`, and `tool_seconds_mean` continue to cover the complete task through repair, recovery, and validation.

Evaluation reads explicit exploration totals from the ledger. It can also derive them from a complete `steps` list whose actions each identify `phase` (`setup`, `acquisition`, or `repair`) and `kind` (`model` or `tool`). When `calls` is supplied, it must equal the step count. `steps_complete=False` marks a partial list. Missing phase information or an incomplete list leaves derived exploration metrics `null`; roles do not determine phases. Recorded exploration call counts and durations remain available when provider token usage is pending.

## Context Selection Policies

| Policy | `Settings.selection` | Selection behavior |
| --- | --- | --- |
| LOCUS | `coverage` | Reserve mandatory edit source, then select additional fact coverage per token and remove redundancy |
| EditOnly | `edit_only` | Retain only source supporting the edit locations |
| StaticTopK | `static_topk` | Append whole source records that fit, in the initial retrieval order |
| ExactDedup | `exact_dedup` | First merge roles and coordinates for identical text hashes in the same initial order, then append context that fits |
| Broad-context selection option | `broad` | Use a larger caller-specified packet capacity with the fixed source order |

Under StaticTopK, original records with identical text consume capacity separately. ExactDedup retains one payload per text hash, including all its roles and coordinates. Both use the fixed initial order. LOCUS recalculates marginal coverage from the currently unmet facts each round. `select_packet(..., raw_sources=..., count=...)` uses the actual tokenizer to recount serialized tokens after role or coordinate metadata changes.

The full broad-context agent baseline uses mini-swe-agent. The built-in `broad` option selects wide context for the LOCUS patching interface. See the project's baseline instructions to run and import full agent baselines.

## Evidence Evaluation

Reference source units and edit source units are separate evaluation inputs supplied only to evaluation functions. Agent inference uses the issue description, base commit, and accessible source code.

Use consistent granularity and exact identities for source units. Supported formats are `path:start:end` strings, `(path, start, end)` triples, and dictionaries containing `path`, `start`, and `end`. Line numbers are one-based and inclusive. `reference_units` contains edit and support units; `edit_units` is its edit subset. `selected_units` comes from the visible evidence packet actually delivered to the patching role.

```python
from locus.evaluation import evidence_metrics

def attach_evidence_metrics(record, selected_units, reference_units, edit_units,
                            admitted_tokens, repeated_tokens, recovered):
    result = dict(record)
    result["evidence_metrics"] = evidence_metrics(
        selected_units=selected_units,
        reference_units=reference_units,
        edit_units=edit_units,
        predicted_files=record["file_predictions"],
        admitted_tokens=admitted_tokens,
        repeated_tokens=repeated_tokens,
        recovery=recovered,
    )
    return result
```

| Metric | Per-attempt denominator or condition | Aggregate field |
| --- | --- | --- |
| File@1 | Whether the first predicted file is a reference edit file; aggregate over the attempt count | `file_at_1_pct` |
| Context recall | Matched units / reference units; requires a nonempty reference set | `recall_pct` |
| Context precision | Matched units / `max(1, retained units)` | `precision_pct` |
| Edit | All reference edit units are retained | `edit_pct` |
| Joint | All reference edit and support units are retained | `joint_pct` |
| Duplication | Repeated source tokens / admitted source tokens | `duplication_mean_pct` |
| Yield | Matched support units / admitted tokens, multiplied by 1,000 | `yield_mean` |
| Recovery | Whether extra source is requested after hand-off, including failed attempts | `recovery_pct` |

The evidence function returns recall, precision, and duplication as fractions; aggregation converts them to percentages. `support_hits` counts only reference support units, excluding reference edit units. Identical source units count once in coverage evaluation. Tokens sent repeatedly are accounted for separately in Duplication.

`recall_pct`, `precision_pct`, `duplication_mean_pct`, and `yield_mean` are means of per-attempt ratios. `pooled_recall_pct`, `pooled_precision_pct`, `pooled_duplication_pct`, and `pooled_yield` pool their respective numerators and denominators before division. `metric_attempts` reports the valid attempt count for each statistic.

Pooled precision is `100 * sum(hits_i) / sum(max(1, selected_i))`; `precision_denominator_total` reports that denominator. The per-attempt denominator floor also applies to attempts that retain no source units.

## Five Repeats and Task-Cluster Intervals

Run `sweep` with the same task roster. All conditions share the batch identifier, benchmark, base commits, model, sampling requests, and outer call and time limits. The root seed defaults to 0. The program derives a 31-bit logical seed from `(root_seed, benchmark, instance_id, repeat)`. `repeat` ranges from 0 to 4, and the logical seed is independent of method and task order. `attempts-design.json` stores the complete run design, and each attempt stores a compact `run_design`. Ablation factors are recorded in `factors`; actual resource consumption is accounted for separately by method.

`seed_mode="off"` uses the local logical seed. OpenAI-format endpoints that support seeds can use `seed_mode="request"`; the program derives request seeds from the logical seed, role, and that role's call index. The actual transmitted seed is recorded in `model_runtime.requests[].requested_seed`. The response model and backend fingerprint are recorded in `served_model` and `system_fingerprint`, respectively.

After officially grading each method independently, pass multiple graded files in one evaluation command:

```bash
locus evaluate --attempts ../locus-work/full-graded.jsonl ../locus-work/ablation-graded.jsonl --task-roster ../locus-work/pro-v2/tasks.jsonl --expected-repeats 5 --bootstrap-samples 2000 --confidence 0.95 --bootstrap-seed 0 --output ../locus-work/comparison.json
```

`--task-roster` defines the complete task set. When omitted, the statistical task set is the union of tasks across methods within each benchmark/model group. Every method is checked against that common roster. `comparisons[].common_roster_complete` records this check; operating-point differences require complete rosters, and the cheapest-baseline comparison requires complete costs for every supplied baseline. Merge official grades separately for each method before comparing methods. The pairing key is `(benchmark, model, instance_id, repeat)`. Duplicate records, invalid repeat indices, and invalid grades raise errors.

`repeats[].counts_by_repeat` lists each run's `successes`, `attempts`, graded/ungraded counts, and missing-task count. For five resolution rates `r_0,...,r_4`, sample variance is `sum((r_i-mean(r))**2)/4`, and sample standard deviation is its square root. Resolution rates are percentages, so variance is measured in squared percentage points. `sample_variance`, `sample_std`, and `mean` are produced when all five task rosters and grades are complete.

For each comparison, `paired_comparisons` reports per-run S/N and four paired counts: both succeed, only the left succeeds, only the right succeeds, and neither succeeds. `resolution_delta_pp` is left resolution minus right resolution. `design_audit` checks the shared batch, tasks, seeds, model, sampling requests, and outer budgets, and lists intentionally varied experimental factors separately. Actual request and backend observations appear in `request_observations`.

Bootstrap resampling treats all repeats of one task as a single task cluster, preserving repeats and method pairs within the cluster. Each replicate uses the same task weights across methods. Percentiles form the nominal 95% interval, and per-run Delta Res intervals use the same task weights. `bootstrap_seed` fixes statistical resampling and is configured separately from the root seed used for execution.

```python
from locus.analysis import resolution_interval, repeat_statistics

one_group = [r for r in records if r["benchmark"] == benchmark
             and r["model"] == model and r["system"] == system]
interval = resolution_interval(one_group, samples=2000, confidence=0.95, seed=0)
repeats = repeat_statistics(one_group, metric="resolution_pct", expected_repeats=range(5))
```

`interval` contains the point estimate, bounds, valid resample count, and task-cluster count. `repeat_statistics` returns per-run counts, metrics, mean, sample variance, and sample standard deviation. `repeat_count` is the number of repeats actually present, `known_repeat_count` is the number with complete metric values, and `five_repeat_complete` checks five-repeat completeness. Missing pairs, missing tasks, or ungraded attempts leave the corresponding differences and intervals `null`.

```python
from locus.analysis import paired_comparison, highest_baseline_comparison

paired = paired_comparison(records, "LOCUS", "Baseline", expected_repeats=range(5),
                           samples=2000, confidence=.95, seed=0)
highest = highest_baseline_comparison(records, "LOCUS", expected_repeats=range(5),
                                     samples=2000, confidence=.95, seed=0)
```

Table 2-style comparisons treat all supplied baselines as candidates. The pooled interval from `highest_baseline_comparison` reselects the highest-resolution baseline in every resample and sets `comparator_reselected=true`. Per-run S/N, Delta Res, and intervals keep the observed pooled comparator fixed and set `conditional_on_comparator=true`. Quality ties select the lower complete mean cost. While costs are pending, tied candidates and the pooled Delta Res/interval remain available. All methods share task-cluster weights, and all five repeats of a task remain in one cluster.

## Budget Allocation and Vote Agreement

Call `allocation_summary` with one record per task containing `instance_id`, `budget_class`, `agreement`, and `votes`. `votes` maps each role to its final valid L/M/H label. The function computes task counts by class, agreement quartiles, the tie rate, and the share of ties promoted to H.

```python
from locus.analysis import (
    allocation_summary, make_agreement_cohorts, agreement_correlation,
)

distribution = allocation_summary(task_records)
cohorts = make_agreement_cohorts(task_records, cohorts=45, per_class=10, seed=0)
correlation = agreement_correlation(task_records, cohorts, samples=2000, seed=0)
```

Correlation inputs additionally contain `resolution`, the task's success fraction across five runs. Each cohort selects ten tasks from each of L, M, and H, then computes Pearson correlation between cohort mean agreement and resolution rate. Tasks may recur across cohorts. Bootstrap resampling reuses one sampling weight for the same task in every cohort.

Uniform-L/M/H fix `Settings.allocation="L"/"M"/"H"`, respectively, and use the same patching and validation workflow as `"adaptive"`. Allocation comparisons pool all repeats over the complete task set.

The allocation sweep runs LOCUS first. For tasks that LOCUS assigns to class d, Uniform-d shares the LOCUS attempt record and its complete-task consumption at the same task and repeat index. Other tasks run with the Uniform-d configuration. Shared records carry `shared_attempt` metadata. Each policy's complete output is still aggregated over all its tasks and repeats. Compare allocation policies using the pooled All results; each shared attempt is counted once within its corresponding policy.

## Candidate Width and Packet Capacity

Candidate widths are `K = 1, 3, 5, 8, 12, 18`. Packet capacities are 4,000, 8,000, 16,000, and 24,000 tokens. Generate the experimental settings from configuration objects:

```python
from dataclasses import replace

width_settings = [replace(base_settings, frontier_size=k)
                  for k in (1, 3, 5, 8, 12, 18)]
capacity_settings = [replace(base_settings, packet_tokens=cap)
                     for cap in (4000, 8000, 16000, 24000)]
uniform_settings = [replace(base_settings, allocation=label)
                    for label in ("L", "M", "H")]
```

Record the current configuration's `frontier_size` in each attempt. Aggregate separately by width to keep configurations in separate result groups.

```python
for k in (1, 3, 5, 8, 12, 18):
    subset = [r for r in one_group if r["frontier_size"] == k]
    print(k, aggregate(subset), repeat_statistics(subset, "tokens_million_mean"))
```

Present the default K=5 reference point separately. Error bars for other widths are calculated from their own five repeats.

## Fixed-edit context interventions

The fixed-edit experiment reuses one established `frontier`, acquired source `pool`, and 16k selected packet `selected`. Each condition uses the same edit locations, resets the patching-role session, and disables recovery reads after hand-off while retaining local validation feedback.

| Condition parameter | Contents | Packet cap |
| --- | --- | --- |
| `broad` | Retain context in acquired-source order | 32k |
| `selected` | The packet chosen by coverage-driven selection | 16k |
| `same-size-random` | Retain mandatory edit source and randomly select whole records to match the Selected packet's actual serialized token count | 16k |
| `required-evidence-removed` | Remove source records for reference support units, excluding mandatory edit source; retain other context within the cap | 12k |
| `duplicates-removed` | Preserve existing fact coverage and mandatory edit source while removing redundant support | 12k |

```python
from locus.interventions import build_interventions

pool_rows = [{**source.as_dict(), "tokens": source.tokens} for source in pool]
selected_rows = [{**source.as_dict(), "tokens": source.tokens} for source in selected]
requirement_rows = [requirement.as_dict() for requirement in requirements]
conditions = build_interventions(
    frontier=[candidate.as_dict() for candidate in frontier],
    pool=pool_rows,
    selected=selected_rows,
    mandatory_ids=mandatory_source_ids,
    reference_support_units=reference_support_units,
    requirements=requirement_rows,
    seed=0,
)
```

Each returned condition contains `frontier`, `observations`, `token_cap`, actual `tokens`, `selection_budget`, and `recovery_allowed=False`. Source records contain `id`, `text`, and `tokens`, with `units` or `locations` identifying the corresponding source units. `tokens` counts actual serialized tokens, including role and coordinate metadata. Selection preserves whole source intervals. The random condition's `selection_budget` equals the Selected packet's actual token count, and the selected whole records sum to exactly that count.

`requirements` uses fact `id` and `source_ids` from the hand-off snapshot as a fixed binary coverage relation. Redundancy removal preserves the covered fact set and all mandatory edit source. Without a fact map, redundancy is determined by source-unit coverage. Removal conditions use a 12k cap and check that the selected source satisfies both the removal rule and capacity.

Pass each condition to the same patching and validation workflow, then aggregate cost, resolution, and reference-evidence metrics from its attempt records. Use `frozen_edit_intervention(condition, ...)` to construct a single condition. The fixed-edit experiment runs all 642 Pro V2 tasks five times, yielding 3,210 attempts per condition.

### Fixed-Edit Command Line

First use `run --packet-dir ../locus-work/packets` to save the visible packet and source pool at hand-off. For one repeat of task `TASK_ID`, construct and execute the Selected condition:

```bash
locus interventions --packet ../locus-work/packets/LOCUS/TASK_ID-r0.json --condition selected --output ../locus-work/conditions
locus repair-frozen --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --instance-id TASK_ID --condition ../locus-work/conditions/selected.json --repeat 0 --system Selected16k --output ../locus-work/selected-r0.jsonl
```

Use `--condition broad`, `same-size-random`, or `duplicates-removed` for other conditions. `required-evidence-removed` takes `--reference-support reference-support.json`, a JSON list of reference support source units. Use `--condition all` with that reference-support input to construct all five conditions. Grade each condition's output through the independent grading entry point for the corresponding public benchmark.

### Evidence Annotation Inputs

`locus evaluate --annotations annotations.jsonl` joins reference source units by `instance_id`. Each line has this structure:

```python
annotation = {
    "instance_id": task_id,
    "reference_units": reference_edit_and_support_units,
    "edit_units": reference_edit_units,
}
```

Evaluation reads `selected_units` from the attempt's actually visible packet by default. When using a different source-unit segmentation, supply `selected_units` at that same granularity in the corresponding evaluation row. The public benchmark's `grading.jsonl` can provide reference edit files for File@1 directly through `--grading-metadata`.
