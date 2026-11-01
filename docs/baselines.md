# Running and importing baselines

## Systems in the paper

| Table 1 system | Official project | Benchmark entry point |
|---|---|---|
| SWE-agent | [SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent) | [run-batch](https://swe-agent.com/latest/usage/batch_mode/) |
| Agentless | [OpenAutoCoder/Agentless](https://github.com/OpenAutoCoder/Agentless) | [Localization, repair, validation, and selection](https://github.com/OpenAutoCoder/Agentless/blob/main/README_swebench.md) |
| OpenHands | [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) | [OpenHands benchmarks](https://github.com/OpenHands/benchmarks) |
| Live-SWE-agent | [OpenAutoCoder/live-swe-agent](https://github.com/OpenAutoCoder/live-swe-agent) | mini-swe-agent with the official `livesweagent.yaml` |
| CCA | [facebookresearch/cca-swebench](https://github.com/facebookresearch/cca-swebench) | Upstream PEX and task container entry points |
| Claude Code | [Claude Code](https://code.claude.com/docs/en/overview) | Pro V2 `locked_claude_code:LockedClaudeCode` |
| Codex | [Codex CLI](https://developers.openai.com/codex/cli) | Pro V2 `locked_codex:LockedCodex` |
| mini-swe-agent / Broad context | [SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | [SWE-bench batch](https://mini-swe-agent.com/latest/usage/swebench/); Pro V2 `locked_mini_swe:LockedMiniSwe` |

Generate patches through the official entry points above, then import them into this project for evaluation. `configs/baselines.toml` provides official links and Lite batch command templates. For a sequence of stages, configure `commands = [[...], [...]]` to execute the argument lists in order. Each argument is passed directly to the process. API keys are read from the corresponding provider environment variables. Install and run Claude Code and Codex according to their official documentation, then import submitted patches through the interface below.

## Lite batch templates

Install the upstream project and run its template from that project's repository directory. This example uses mini-swe-agent:

```python
from pathlib import Path
from locus.baselines import run_external

run_external(
    Path('/path/to/LOCUS-ICDE/configs/baselines.toml'), 'mini-swe-agent',
    {'model': 'openai/gpt-5.2', 'workers': 1, 'benchmark': 'lite',
     'output': '/path/to/locus-work/mini-lite-r0'},
    cwd=Path('/path/to/mini-swe-agent'))
```

For SWE-agent, select the `swe-agent` template. For Live-SWE-agent, follow the [official setup instructions](https://github.com/OpenAutoCoder/live-swe-agent#-setup) to install mini v1 (`pip install 'mini-swe-agent~=1.0'`). In a workspace checked out at the task's base commit, run `mini --config /path/to/live-swe-agent/config/livesweagent.yaml`, then export the workspace's Git patch.

Run OpenHands from its official `benchmarks` repository. First prepare the SDK and task images according to the [SWE-bench instructions](https://github.com/OpenHands/benchmarks/blob/main/benchmarks/swebench/README.md):

```bash
uv run python -m benchmarks.swebench.build_images --dataset princeton-nlp/SWE-bench_Lite --split test --image ghcr.io/openhands/eval-agent-server --target source-minimal
```

Select the `openhands` template and supply `llm_config` in the context as an absolute path to the upstream LLM JSON configuration. Convert the resulting `output.jsonl` into standard patch predictions with the upstream command:

```bash
uv run swebench-eval /path/to/openhands/output.jsonl --dataset princeton-nlp/SWE-bench_Lite --output-file /path/to/locus-work/openhands-r0-preds.jsonl --skip-evaluation
```

Use separate output directories `r0` through `r4` for the five repeats. Set the paper's parameters in each upstream configuration: temperature 0.7, top-p 1.0, 128,000 input tokens, 32,768 output tokens, 250 model/tool calls, 8 vCPUs, and 32 GiB. The task time limit is 7,200 seconds for Lite/V1 and 3,000 seconds for V2. Sonnet 4.5 requests send temperature only; record `top_p` as omitted and the provider default as `0.99`. See [upstream sampling settings](upstream_sampling.md) for template overrides and model-specific configuration.

## Importing predictions

The reader accepts upstream `preds.json`, a JSON array, or `all_preds.jsonl`:

```python
from locus.baselines import load_predictions, import_predictions
from locus.benchmarks import read_jsonl, write_jsonl

tasks = read_jsonl('../locus-work/lite/tasks.jsonl')
predictions = load_predictions('../locus-work/mini-lite-r0/preds.json')
records = import_predictions(predictions, tasks, system='mini-swe-agent',
                             model='gpt-5.2', repeat=0)
write_jsonl('../locus-work/mini-lite-r0-attempts.jsonl', records)
```

For Agentless, import the final reranked `all_preds.jsonl`. For OpenHands, import the standard JSONL produced by the conversion above. For SWE-agent and mini-swe-agent, import `preds.json`. For Live-SWE-agent, CCA, Claude Code, and Codex, write final patches as JSONL with one object per line containing `instance_id` and `model_patch`. Empty or null patches remain failed submissions. Import each system, model, version, and repeat separately, then follow [datasets and evaluation](benchmarks.md) to grade the patches.

Supply complete resource measurements for each attempt through `telemetry={instance_id: {'ledger': ...}}`. Normalized cost uses the token and worker prices in the paper's Eq. (5). Import patch predictions and resource measurements separately; the grader fills in resolution status.

## Official Pro V2 system entry points

Configure model environment variables in the official Pro harness, run the corresponding system through its locked wrapper, then use the official `patch_replay:PatchReplayAgent`. See the [official V2 README](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md) for command arguments. This project also provides `export_harbor_replay` to export final patches in the format accepted by that replay agent.

## Context selection baselines and component experiments

The paper's EditOnly, StaticTopK, ExactDedup, and LOCUS methods share the repair and validation workflow. They differ in the evidence selected for the repair role. For fixed-edit experiments, freeze candidate edits first, choose the Broad32k, Selected16k, Same-size random, Required-evidence removed, or Duplicates removed condition, and disable recovery reads. Selection functions are in `locus.interventions`; `locus.evaluation` and `locus.analysis` compute the statistics.
