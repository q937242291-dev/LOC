# Public Datasets and Evaluation

## Datasets

| Benchmark | Public dataset | Dataset configuration | Test tasks | Five repeats |
|---|---|---|---:|---:|
| `lite` | [SWE-bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) | `test` | 300 | 1,500 |
| `pro-v1` | [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) | `v1`, revision `v2.0.0` | 731 | 3,655 |
| `pro-v2` | [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) | `default`, revision `v2.0.0` | 642 | 3,210 |

The Pro `v1` configuration contains the original release; `default` selects V2. Prepare and grade the two releases separately.

```bash
pip install -e '.[benchmarks]'
python -c "from pathlib import Path; from locus.benchmarks import fetch_public; fetch_public('lite', Path('../locus-work/lite'))"
python -c "from pathlib import Path; from locus.benchmarks import fetch_public; fetch_public('pro-v1', Path('../locus-work/pro-v1'))"
python -c "from pathlib import Path; from locus.benchmarks import fetch_public; fetch_public('pro-v2', Path('../locus-work/pro-v2'))"
```

Each directory receives `tasks.jsonl` and `grading.jsonl`. The agent reads `tasks.jsonl`, which contains the `instance_id`, repository, base commit, issue and requirements, image, and working directory. Independent evaluation reads `grading.jsonl`. The download commands retrieve the public data.

`repeat_plan(tasks, repeats=5, seed=0)` creates five attempts per task. The `repeat` index ranges from 0 to 4. Export and grade each repeat separately, then pool the attempts. Logical seeds are derived with SHA-256 from the root seed, benchmark, task ID, and repeat index, retaining 31 bits. They are independent of task order, subsets, and method names. The `--seed` option for `run` and `sweep` sets the root seed, which defaults to 0. The `[model].seed_mode` setting controls whether provider API seeds are sent; actual request settings are recorded separately.

## Lite Evaluation

Install the [official SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/) and Docker. Export one repeat in the official prediction format:

```python
from pathlib import Path
from locus.benchmarks import read_jsonl, export_predictions

records = read_jsonl('../locus-work/attempts.jsonl')
export_predictions(records, Path('../locus-work/lite-r0.jsonl'), repeat=0)
```

```bash
python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Lite --split test --predictions_path ../locus-work/lite-r0.jsonl --max_workers 1 --run_id locus-lite-r0
```

Grades come from the official test results. Pass the harness summary JSON path, or its individual run directory, to `read_lite_grades(report_path, repeat=0)`. It reads both per-task `report.json` files and aggregate `results.json` files. Entries in the official summary's `empty_patch_ids` count as failed submissions; incomplete verification remains ungraded. Use `merge_grades(records, grades)` to join grades to attempt records. Use a separate `run_id` for each repeat.

## Pro V1 Evaluation

Obtain the [official Pro harness](https://github.com/scaleapi/SWE-bench_Pro-os), install its dependencies, and pass the exported JSON array to its V1 evaluation entry point:

```python
from pathlib import Path
import subprocess
from locus.benchmarks import (read_jsonl, export_pro_v1_patches,
                              pro_v1_grade_command)

records = read_jsonl('../locus-work/attempts.jsonl')
patches = Path('../locus-work/pro-v1-r0.json')
export_pro_v1_patches(records, patches, repeat=0)
command = pro_v1_grade_command(
    Path('../SWE-bench_Pro-os'), Path('../locus-work/pro-v1/grading.jsonl'),
    patches, Path('../locus-work/pro-v1-r0-grade'), workers=1)
subprocess.run(command, cwd=Path('../SWE-bench_Pro-os').resolve(), check=True)
```

Read the official output with `read_pro_v1_grades(Path('../locus-work/pro-v1-r0-grade/eval_results.json'), repeat=0)`.

## Pro V2 Evaluation

The [official V2 tasks and protocol](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md) provide a public image, `task.toml`, and verification scripts for each task. Prepare the upstream checkout:

```bash
git clone --branch v2.0.0 https://github.com/scaleapi/SWE-bench_Pro-os.git ../SWE-bench_Pro-os
```

`grade_pro_v2` applies the submitted patch in a fresh container, runs the task's original `tests/test.sh`, and reads `/logs/verifier/reward.txt`:

```python
from pathlib import Path
from locus.benchmarks import read_jsonl, write_jsonl, grade_pro_v2, merge_grades

records = read_jsonl('../locus-work/attempts.jsonl')
tasks = Path('../SWE-bench_Pro-os/v2/tasks')
grades = [grade_pro_v2(tasks / row['instance_id'], row) for row in records]
write_jsonl('../locus-work/graded.jsonl', merge_grades(records, grades))
```

Repair runs in an offline repository container, while the controller process sends model requests. Repositories usually reside in `/app`; container startup checks the Git working directory and automatically recognizes tasks that use `/testbed`. Verification applies the submitted patch in a separate fresh image and permits dependency downloads under the official protocol. Use `grade_pro_v2(..., network='none')` for offline verification when dependencies are already installed. Verification uses the task's original `tests/test.sh`, resource settings, and verification timeout from `task.toml`. Image preparation and patch application are separate from the verification timeout. Repair uses the paper's configuration of 8 vCPUs, 32 GiB, and a 3,000-second task limit.

You can also export patches for the official Harbor replay in fresh images:

```python
from pathlib import Path
from locus.benchmarks import export_harbor_replay, read_jsonl

export_harbor_replay(read_jsonl('../locus-work/attempts.jsonl'),
                     Path('../locus-work/replay-r0'), repeat=0)
```

Run from the official harness directory:

```bash
PYTHONPATH=v2/tooling harbor run -p v2/tasks -e modal -n 1 -a patch_replay:PatchReplayAgent -m replay --ak source_job=../locus-work/replay-r0 --job-name locus-pro-v2-r0-regrade
```

For every release, generate patches first, then use the corresponding independent evaluation harness to assign `resolved`. Pool scores across all tasks and repeats, including failed attempts in the cost totals.
