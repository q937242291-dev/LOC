# LOCUS

**Think as Needed, Know When to Stop: An Adaptive Resource-Control Agent for Software Engineering**

LOCUS connects evidence acquisition, source reuse, exploration budgets, patch generation, and local validation in a resource-control loop for repository repair. The issue and repository structure determine the exploration profile. Role-aware grounding forms candidate edits and support requirements; coverage gain per token selects the evidence packet; four evidence conditions govern the handoff to repair.

The project includes the agent, public benchmark interfaces, evaluation metrics, baseline integration, and experiment configurations. Public datasets and models are available through the links below. Choose an external working directory for downloads and run outputs.

## 1. Installation

Use Python 3.11+, Git, and Docker with Linux containers. Each repair worker uses 8 vCPUs and 32 GiB of memory. On Windows, use Docker Desktop in Linux container mode.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[benchmarks,tokenizers]"
locus --help
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[benchmarks,tokenizers]"
```

For GPT-5.2 or Sonnet, install `python -m pip install -e ".[benchmarks]"`. The DeepSeek configuration loads its public tokenizer from Hugging Face.

## 2. Models and credentials

| Paper backbone | Configuration | Official links |
|---|---|---|
| DeepSeek-V4-Pro | `configs/deepseek.toml` | [Model and weights](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [API](https://api-docs.deepseek.com/) |
| Claude Sonnet 4.5 | `configs/sonnet.toml` | [Model](https://www.anthropic.com/news/claude-sonnet-4-5), [API](https://platform.claude.com/docs/en/api/messages/create) |
| GPT-5.2 | `configs/gpt52.toml` | [Model and API](https://developers.openai.com/api/docs/models/gpt-5.2) |

All three configurations read credentials from `LOCUS_API_KEY`. Enter the key in the current Linux/macOS terminal:

```bash
read -rs -p "API key: " LOCUS_API_KEY
export LOCUS_API_KEY
```

PowerShell 7:

```powershell
$env:LOCUS_API_KEY = Read-Host -MaskInput "API key"
```

Configuration files contain the model identifier, endpoint, and credential environment variable name. Credentials stay in the controller process; repository workers run in offline containers. `.env.example` contains an empty configuration. The Git ignore rules cover local credentials, downloads, and run outputs.

For a self-hosted public model, set `provider = "compatible"`, `base_url`, and `model`. The endpoint supplies OpenAI-format `/chat/completions` responses with token usage.

## 3. Public datasets

| Benchmark | Public dataset | Configuration | Tasks | Five attempts per task |
|---|---|---|---:|---:|
| SWE-bench Lite | [Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) | `test` | 300 | 1,500 |
| SWE-bench Pro V1 | [Dataset](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) | `v1`, revision `v2.0.0` | 731 | 3,655 |
| SWE-bench Pro V2 | [Dataset](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) | `default`, revision `v2.0.0` | 642 | 3,210 |

Download into a working directory outside the project:

```bash
locus fetch --benchmark lite --output ../locus-work/lite
locus fetch --benchmark pro-v1 --output ../locus-work/pro-v1
locus fetch --benchmark pro-v2 --output ../locus-work/pro-v2
```

The downloader creates `tasks.jsonl` for inference and `grading.jsonl` for independent evaluation. Inference inputs include the issue, requirements, repository, base commit, image, and working directory. Workers start from the corresponding public image and read source at the base commit.

See [public benchmark usage](docs/benchmarks.md) for dataset versions, images, grading, and the five-repeat workflow.

## 4. Running LOCUS

Start with one attempt on one task:

```bash
locus run --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --limit 1 --repeats 1 --output ../locus-work/smoke/attempts.jsonl
```

Run the complete Pro V2 workload:

```bash
locus run --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --output ../locus-work/pro-v2-deepseek/attempts.jsonl
```

Select another backbone with `--config`, or select Lite or Pro V1 with `--tasks`. Use `--instance-id` for a specific task and `--resume` to continue an existing task/repeat plan.

Each output includes the final `model_patch`, candidate files, gate state, budget votes, complete-task resource accounting, and local validation status. The corresponding benchmark evaluator supplies the official success result.

### Main parameters

| Parameter | Value |
|---|---|
| Sampling request | temperature 0.7, top-p 1.0 |
| Request input / output limit | 128,000 / 32,768 tokens |
| Complete-task call limit | 250 model and tool calls |
| Evidence packet / candidate width | 16,000 tokens / K=5 |
| Debate | Five roles, three rounds; the last valid vote from each role |
| L / M / H | 64 / 128 / 210 exploration calls; relation depth 1 / 2 / 3 |
| Ties | Choose the higher profile; choose H when all votes are invalid |
| Task deadline | Pro V1: 7,200 seconds; Pro V2: 3,000 seconds |
| Grounding | lambda=0.78, mu=0.10, alpha=(0.50, 0.25, 0.25) |

`--packet-tokens`, `--frontier-size`, `--allocation`, `--window`, and `--deadline` override the corresponding settings. Other parameters are in the TOML `[agent]` section. Attempt records contain the effective configuration.

`stability_window` controls the consecutive identical ordered-frontier window. `reserve_calls` and `reserve_seconds` reserve resources for patching and local validation. Project defaults are 2, 40 calls, and 300 seconds, respectively. Lite uses a default deadline of 7,200 seconds, adjustable with `--deadline`.

DeepSeek evidence serialization uses its public tokenizer. GPT and Sonnet evidence serialization uses `o200k_base`. Billed token counts come from the provider's uncached input, cached input, and output usage. The DeepSeek configuration disables thinking to activate temperature 0.7. GPT-5.2 uses `reasoning_effort="none"`.

The native Sonnet 4.5 adapter sends `temperature=0.7` and omits `top_p`. Its interface accepts one sampling control; the published Messages API default for `top_p` is 0.99. Runtime records contain both requested values and transmitted fields. See the [sampling interface](https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/model-migration.md#sampling-parameters---temperature--top_p--top_k) and [provider default](https://platform.claude.com/docs/en/release-notes/overview#may-22-2025).

### Repeats, seeds, and paired runs

Each task has `repeat` values 0 through 4. The root seed defaults to 0 and can be set with `--seed`. The attempt seed is derived from the root seed, benchmark, task ID, and repeat. Method names, task ordering, and task subsets preserve the paired seed of a given task and repeat.

`run` and `sweep` create an `attempts-design.json` file next to the attempt output. It records the task roster digest, base commits, code version, model, sampling request, and budgets. All configurations in one sweep share a batch ID; `--batch-id` selects an explicit experiment batch. Each attempt stores compact `run_design` metadata. Resume checks the batch, tasks, seeds, settings, and code version.

`[model].seed_mode` defaults to `"off"`. Set `"request"` for an OpenAI-format endpoint with seed support. API seeds are derived from the attempt seed, role, and role-local call index. The native Anthropic adapter uses `"off"`. `model_runtime.requests` records transmitted sampling fields, API seeds, returned model identifiers, and backend fingerprints. Parameter syntax is documented in [OpenAI Docs](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create#seed).

## 5. Independent evaluation

### Lite

Install the [official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/), then export and grade each repeat:

```bash
locus export --attempts ../locus-work/lite-deepseek/attempts.jsonl --repeat 0 --output ../locus-work/lite-r0.jsonl
locus grade-lite --predictions ../locus-work/lite-r0.jsonl --run-id locus-lite-r0 --workers 1 --execute
locus read-grades --format lite --input logs/run_evaluation/locus-lite-r0 --repeat 0 --output ../locus-work/lite-grades-r0.jsonl
```

Repeat for IDs 0 through 4. Running `grade-lite` without `--execute` prints the official command arguments.

### Pro V1 / V2

Obtain the [official Pro harness](https://github.com/scaleapi/SWE-bench_Pro-os). For V1, use `locus export --format pro-v1` and `locus grade-pro-v1`. For V2, replay patches in fresh official images:

```bash
locus grade-pro-v2 --attempts ../locus-work/pro-v2-deepseek/attempts.jsonl --tasks-dir ../SWE-bench_Pro-os/v2/tasks --output ../locus-work/pro-v2-grades.jsonl
locus merge-grades --attempts ../locus-work/pro-v2-deepseek/attempts.jsonl --grades ../locus-work/pro-v2-grades.jsonl --output ../locus-work/pro-v2-graded.jsonl
locus evaluate --attempts ../locus-work/pro-v2-graded.jsonl --grading-metadata ../locus-work/pro-v2/grading.jsonl --output ../locus-work/pro-v2-summary.json
```

The V2 verifier uses its original `tests/test.sh` and `reward.txt`. Predictions can also be exported for the official Harbor `PatchReplayAgent`. See [benchmark evaluation](docs/benchmarks.md) for complete commands.

### Metrics

`evaluate` aggregates Resolution, File@1, total tokens, tokens per success, complete-task time, normalized cost, cost per success, cached-input share, and post-handoff recovery. With `--annotations`, it also computes reference-context recall and precision, Edit, Joint, Duplication, and Yield. `tokens_per_success` divides the complete-task tokens of all attempts, including failures, by the number of successful attempts; `tokens_million_per_success` reports the same value in millions.

Table 5(a) exploration diagnostics use `exploration_calls_mean` and `exploration_llm_seconds_mean`, counting setup and acquisition before handoff. `exploration_model_calls_mean` and `exploration_tool_calls_mean` give the two call categories separately. `calls_mean` and `llm_seconds_mean` cover the complete task, including repair and recovery.

Normalized cost follows Eq. (5):

```text
USD = 1e-6 * (1.25 * uncached_input + 0.10 * cached_input + 6.00 * output)
      + 0.06 * worker_hours
```

Cost per success divides the cost of all attempts by the number of successful attempts, including failed attempts in the numerator. Source-unit definitions, input formats, denominators, task-cluster bootstrap, and vote diagnostics are documented in [evaluation and analysis](docs/evaluation.md).

### Per-run counts, resolution differences, and five-repeat variance

Grade each method independently, then pass its graded attempts to `evaluate`. For example, compare the full method with a component ablation:

```bash
locus evaluate --attempts ../locus-work/full-graded.jsonl ../locus-work/without-gate-graded.jsonl --task-roster ../locus-work/pro-v2/tasks.jsonl --expected-repeats 5 --bootstrap-samples 2000 --bootstrap-seed 0 --confidence 0.95 --output ../locus-work/component-summary.json
```

`repeats` reports successes S, attempts N, and resolution per run, followed by the five-repeat mean, sample variance, and sample standard deviation. Resolution uses percentages; variance uses squared percentage points and denominator 4. `paired_comparisons` includes per-run paired contingency counts, resolution differences, 95% intervals, and checks of batch, seeds, model, and budget. `--task-roster` supplies the complete roster for attempt-completeness checks.

Bootstrap resamples tasks as clusters, retaining all repeats and using shared task weights across methods. Pass all graded baseline files when comparing the highest-resolution baseline. `highest_baseline_comparisons` reselects that baseline in each bootstrap sample; per-run counts use the observed pooled comparator. Pending grades and incomplete pairings use `null` for the corresponding statistics.

Methods are compared on a common task roster within each benchmark/model group. The cheapest-baseline comparison uses complete costs from every supplied baseline.

## 6. Baselines

Official entries and usage for the eight system baselines are listed in [baseline integration](docs/baselines.md) and `configs/baselines.toml`:

SWE-agent, Agentless, OpenHands, Live-SWE-agent, CCA, Claude Code, Codex, and mini-swe-agent.

Install and run [Codex](https://developers.openai.com/codex/cli) and [Claude Code](https://code.claude.com/docs/en/overview) through their official documentation. Use the official model and weight links in Section 2.

Import each upstream system's final patches, then grade with the corresponding benchmark:

```bash
locus import-baseline --predictions ../locus-work/mini-lite-r0/preds.json --tasks ../locus-work/lite/tasks.jsonl --system mini-swe-agent --model gpt-5.2 --repeat 0 --output ../locus-work/mini-lite-r0-attempts.jsonl
```

`--telemetry` imports the complete-task resource statistics from that upstream run. `locus.baselines.run_external` executes the configured upstream command templates.

The paper uses a common sampling protocol. See [upstream agent sampling](docs/upstream_sampling.md) for the pinned versions, original temperatures, stage settings, seed interfaces, and official source links. Runs using original upstream settings use separate configuration identifiers, output directories, and summaries.

EditOnly, StaticTopK, ExactDedup, and LOCUS use the project's source-selection implementations with the same patching and validation interface:

```bash
locus sweep --study selection --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --output ../locus-work/selection
```

The paper's Broad context baseline uses upstream mini-swe-agent and the common import interface. The `selection` sweep runs the four source-selection policies above.

## 7. Ablations and sensitivity studies

```bash
locus sweep --study components --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --output ../locus-work/components
locus sweep --study allocation --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --output ../locus-work/allocation
locus sweep --study width --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --reference-attempts ../locus-work/pro-v2-deepseek/attempts.jsonl --output ../locus-work/width
locus sweep --study capacity --config configs/deepseek.toml --tasks ../locus-work/pro-v2/tasks.jsonl --repeats 5 --output ../locus-work/capacity
```

- `components`: The full method and six removals: role grounding, alternatives, gate, memory, packet, and packet/memory.
- `allocation`: Adaptive and Uniform-L/M/H.
- `width`: K=1/3/5/8/12/18. K=5 reuses the main LOCUS attempts supplied through `--reference-attempts`; the other widths run independently. Shared records retain the complete-task ledger, seed, model runtime, and original run design in `shared_attempt`.
- `capacity`: 4k/8k/16k/24k packets and Aggressive/Balanced/Strict stability windows of 1/2/3, with all four gate conditions active.

Full-method and ablation runs share tasks, repeat IDs, logical seeds, model, sampling request, the 250-call outer limit, task deadline, and request input/output ceilings. Packet capacity, candidate width, gate window, and allocation follow the study factor; resource consumption is measured for each condition. The full-method name in `components` is `LOCUS`. Use `--locus-system LOCUS-16k-Balanced` as the capacity reference and `--locus-system LOCUS-K5` as the width reference.

`--plan-only` writes configurations. `locus matrix --output ../locus-work/matrix.json` writes the 81 combinations of three benchmarks, three backbones, and nine systems.

Fixed-edit experiments save the visible packet and acquired source pool with `run --packet-dir`, construct conditions with `locus interventions`, and run them with `locus repair-frozen`. They fix the frontier, disable recovery reads, and retain local validation feedback while comparing Broad32k, Selected16k, same-size random, required-evidence removal, and duplicate removal. See [fixed-edit interventions](docs/evaluation.md#fixed-edit-context-interventions) for the interfaces and reference source-unit inputs.



The five debate roles inspect implementation anchors, dependencies, behavior, alternative hypotheses, and validation requirements. Grounding forms candidate edits and required facts. The Trajectory Analysis Agent summarizes unmet facts, and the Evolving-Planner adjusts read targets. The patching and validation role uses structured `apply_patch`, `test`, `read`, and `finish` actions.

Each role has an independent conversation. New source citations are checked against the source consumed by that role's current request. A handoff snapshot contains the evidence packet, requirements, alternatives, and rank history. Rank stability compares ordered edit locations and intended behavior. Source coordinates and final patches use the fixed base commit. Custom adapters implement `ModelProtocol`, `WorkspaceProtocol`, and `RecipientProtocol`.
