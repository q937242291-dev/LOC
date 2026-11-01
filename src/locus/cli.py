"""Command-line interfaces for preparation, repair, independent grading and analysis."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import subprocess
import sys

from .benchmarks import (BENCHMARKS, export_predictions, export_pro_v1_patches,
                         fetch_public, grade_pro_v2, lite_grade_command, merge_grades,
                         pro_v1_grade_command, read_jsonl, read_lite_grades,
                         read_pro_v1_grades, repeat_plan, write_jsonl)
from .config import Settings, load_config
from .evaluation import aggregate, evidence_metrics
from .types import compact


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def _execute(command, execute, *, cwd=None):
    if execute:
        subprocess.run(command,check=True,**({"cwd":cwd} if cwd is not None else {}))
    else:
        print(compact(dict(argv=command)))


def _selected_tasks(args):
    tasks = read_jsonl(args.tasks)
    if args.instance_id:
        tasks = [t for t in tasks if t["instance_id"] == args.instance_id]
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        raise ValueError("No selected tasks")
    if len({t["instance_id"] for t in tasks}) != len(tasks):
        raise ValueError("Run each benchmark separately with unique instance_id values")
    return tasks


def _width_reference(args, settings, model):
    """Select the main-table K=5 attempts with identical task and runtime controls."""
    from .run_design import attempt_design, build_run_manifest
    if not args.reference_attempts:
        raise ValueError("Width execution requires --reference-attempts from the main LOCUS run")
    tasks = _selected_tasks(args)
    reference_settings = replace(settings, frontier_size=5).validate()
    deadlines = {task.get("benchmark", "custom"): args.deadline if args.deadline is not None else
                 BENCHMARKS.get(task.get("benchmark"), {}).get("deadline", settings.deadline_seconds)
                 for task in tasks}
    manifest = build_run_manifest(tasks, reference_settings, model, repeats=args.repeats,
        reference_settings=settings, deadline_by_benchmark=deadlines)
    expected = {(task["instance_id"], task["repeat"]): task
                for task in repeat_plan(tasks, args.repeats, settings.seed)}
    references = {}
    for row in read_jsonl(args.reference_attempts):
        key = row.get("instance_id"), row.get("repeat")
        if type(key[1]) is not int or key not in expected:
            continue
        if key in references:
            raise ValueError("Main reference contains duplicate task/repeat attempts")
        task = expected[key]
        actual = replace(reference_settings, seed=task["seed"],
                         deadline_seconds=deadlines[task.get("benchmark", "custom")])
        if (row.get("benchmark") != task.get("benchmark", "custom")
                or row.get("model") != model.model
                or type(row.get("seed")) is not int or row["seed"] != task["seed"]
                or not isinstance(row.get("system"), str) or not row["system"].strip()
                or not isinstance(row.get("ledger"), dict)
                or not isinstance(row.get("settings"), dict)):
            raise ValueError("Main reference attempt identity or complete-task controls do not match")
        if asdict(Settings(**row["settings"]).validate()) != asdict(actual):
            raise ValueError("Main reference settings differ from the K=5 width controls")
        metadata = row.get("run_design")
        required = attempt_design(manifest, task, actual)
        compared = ("schema_version", "code_version", "seed_policy", "root_seed", "task", "repeat",
                    "logical_seed", "model", "sampling_requested", "outer_budget", "runtime_controls",
                    "factors", "condition_id")
        if not isinstance(metadata, dict) or any(metadata.get(field) != required[field]
                                                 for field in compared):
            raise ValueError("Main reference run design differs from the K=5 width controls")
        references[key] = row
    if set(references) != set(expected):
        raise ValueError("Main reference lacks a selected task/repeat attempt")
    return references


def _run_attempts(args, settings, model, *, output=None, system=None, reuse_from=None,
                  batch_id=None, reference_settings=None, width_reference=None):
    from .controller import run_agent
    from .run_design import attempt_design, build_run_manifest, validate_manifest
    tasks = _selected_tasks(args)
    target = Path(output or args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    existing = read_jsonl(target) if target.exists() and args.resume else []
    if target.exists() and not args.resume:
        raise ValueError("Output already exists; choose a new output or use --resume")
    name = system or args.system
    manifest_path = target.with_name(target.stem + "-design.json")
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if args.resume and manifest_path.exists() else None
    if existing and previous is None:
        raise ValueError("Resume requires the original run design file")
    if previous is not None:
        validate_manifest(previous)
    requested_batch = batch_id or args.batch_id
    if previous is not None and requested_batch and requested_batch != previous["batch_id"]:
        raise ValueError("Resume batch_id differs from the original run")
    deadlines = {t.get("benchmark", "custom"): args.deadline if args.deadline is not None else
                 BENCHMARKS.get(t.get("benchmark"), {}).get("deadline", settings.deadline_seconds)
                 for t in tasks}
    manifest = build_run_manifest(tasks, settings, model, repeats=args.repeats,
        batch_id=requested_batch or (previous["batch_id"] if previous else None),
        reference_settings=reference_settings, deadline_by_benchmark=deadlines)
    validate_manifest(manifest)
    if previous is not None and any(previous[key] != manifest[key]
                                    for key in ("design_id", "condition_id")):
        raise ValueError("Resume controls, code, task roster or condition changed")
    plan = repeat_plan(tasks,args.repeats,settings.seed)
    planned = {(t["instance_id"], t["repeat"]):t for t in plan}
    done = set()
    for result in existing:
        key = result["instance_id"], result.get("repeat",0)
        if key in done or key not in planned:
            raise ValueError("Resume contains duplicate or unplanned attempts")
        task = planned[key]
        actual = replace(settings, deadline_seconds=deadlines[task.get("benchmark", "custom")],
                         seed=task["seed"])
        if result.get("run_design") != attempt_design(manifest, task, actual):
            raise ValueError("Resume attempt metadata differs from its run design")
        done.add(key)
    _json(manifest_path, manifest)
    total = len(plan)
    with target.open("a" if args.resume else "w",encoding="utf-8") as stream:
        for position,task in enumerate(plan,1):
            if (task["instance_id"],task["repeat"]) in done:
                continue
            deadline = deadlines[task.get("benchmark", "custom")]
            attempt_settings = replace(settings,deadline_seconds=deadline,seed=task["seed"])
            packet_output = None
            if args.packet_dir:
                packet_output = Path(args.packet_dir)/name/(
                    task["instance_id"]+f"-r{task['repeat']}.json")
            key = task["instance_id"], task["repeat"]
            shared = (width_reference if width_reference is not None else reuse_from or {}).get(key)
            if width_reference is not None:
                result = deepcopy(shared)
                result["shared_attempt"] = dict(system=shared["system"],
                    instance_id=task["instance_id"], repeat=task["repeat"],
                    source=str(Path(args.reference_attempts).resolve()),
                    run_design=deepcopy(shared["run_design"]))
            elif (settings.allocation in {"L", "M", "H"} and shared is not None
                    and shared.get("budget_class") == settings.allocation):
                # The allocation study shares the same complete trajectory when
                # adaptive and uniform policies select the identical profile.
                result = deepcopy(shared)
                result["system"] = name
                result["settings"] = {**result.get("settings", {}),
                                      "allocation":settings.allocation}
                result["shared_attempt"] = dict(system=shared["system"],
                    instance_id=task["instance_id"], repeat=task["repeat"])
                if packet_output is not None:
                    source_packet = Path(args.packet_dir)/shared["system"]/packet_output.name
                    if source_packet.is_file():
                        packet_output.parent.mkdir(parents=True, exist_ok=True)
                        packet_output.write_bytes(source_packet.read_bytes())
            else:
                result = run_agent(dict(task,system=name),attempt_settings,model,packet_output=packet_output)
            result.update(benchmark=task.get("benchmark", "custom"), model=model.model,
                          system=name, seed=task["seed"],
                          run_design=attempt_design(manifest, task, attempt_settings))
            stream.write(compact(result)+"\n")
            stream.flush()
            print(f"{position}/{total}: {task['instance_id']} r{task['repeat']} submitted",flush=True)
    return manifest


def _evaluate(args):
    records = [row for path in args.attempts for row in read_jsonl(path)]
    if args.grades:
        if len({(r.get("benchmark"), r.get("model"), r.get("system")) for r in records}) > 1:
            raise ValueError("Merge official grades for each method separately before comparing methods")
        records = merge_grades(records,read_jsonl(args.grades))
    annotations = {r["instance_id"]:r for r in read_jsonl(args.annotations)} if args.annotations else {}
    edit_files = {}
    if args.grading_metadata:
        for row in read_jsonl(args.grading_metadata):
            # Only evaluation reads the public reference diff.
            patch = row.get("patch",row.get("gold_patch",""))
            edit_files[row["instance_id"]] = set(re.findall(
                r"(?m)^(?:\+\+\+ b/|--- a/)(.+)$",patch))
    for record in records:
        ann = annotations.get(record["instance_id"])
        current = record.get("evidence_metrics") or {}
        sources = (record.get("packet") or {}).get("sources",[])
        selected = [loc for src in sources for loc in src.get("locations",[])]
        if ann is not None:
            calculated = evidence_metrics(
                ann.get("selected_units",selected),ann.get("reference_units"),ann.get("edit_units"),
                record.get("file_predictions") or (),
                admitted_tokens=current.get("admitted_tokens"),
                repeated_tokens=current.get("repeated_tokens"),
                recovery=current.get("recovery",record.get("recovery_requested")),
            )
        else:
            calculated = evidence_metrics(
                selected,None,None,record.get("file_predictions") or (),
                admitted_tokens=current.get("admitted_tokens"),
                repeated_tokens=current.get("repeated_tokens"),
                recovery=current.get("recovery",record.get("recovery_requested")),
            )
        record["evidence_metrics"] = {**current,**{
            key:value for key,value in calculated.items()
            if value is not None and (ann is not None or sources or key not in {
                "selected","hits","required","support_hits"})}}
        if record["instance_id"] in edit_files:
            files = record.get("file_predictions") or []
            record["evidence_metrics"]["file_at_1"] = bool(
                files and files[0] in edit_files[record["instance_id"]])
    from .reporting import analysis_report
    rosters = None
    if args.task_roster:
        rosters = {}
        for path in args.task_roster:
            for task in read_jsonl(path):
                rosters.setdefault(task.get("benchmark", "custom"), []).append(task["instance_id"])
    _json(args.output, analysis_report(records, locus_system=args.locus_system,
        expected_repeats=args.expected_repeats, samples=args.bootstrap_samples,
        confidence=args.confidence, seed=args.bootstrap_seed, task_rosters=rosters))
    if args.graded_output:
        write_jsonl(args.graded_output,records)


def parser():
    p = argparse.ArgumentParser(prog="locus",description="LOCUS repository repair and evaluation")
    sub = p.add_subparsers(dest="command",required=True)
    fetch = sub.add_parser("fetch",help="Download a public benchmark into a chosen working directory")
    fetch.add_argument("--benchmark",choices=BENCHMARKS,required=True)
    fetch.add_argument("--output",required=True)
    fetch.add_argument("--revision")
    fetch.add_argument("--limit",type=int)
    def run_options(run, *, output=True):
        run.add_argument("--config",required=True)
        run.add_argument("--tasks",required=True)
        if output:
            run.add_argument("--output",required=True)
        run.add_argument("--repeats",type=int,default=5)
        run.add_argument("--seed",type=int,help="Root seed for the paired task/repeat plan")
        run.add_argument("--batch-id",help="Common execution batch for compared configurations")
        run.add_argument("--instance-id")
        run.add_argument("--limit",type=int)
        run.add_argument("--deadline",type=float)
        run.add_argument("--resume",action="store_true")
        run.add_argument("--packet-dir")
    run = sub.add_parser("run",help="Run LOCUS on public inference tasks")
    run_options(run)
    run.add_argument("--system",default="LOCUS")
    run.add_argument("--selection",choices=("coverage","edit_only","static_topk","exact_dedup","broad"))
    run.add_argument("--allocation",choices=("adaptive","L","M","H"))
    run.add_argument("--packet-tokens",type=int)
    run.add_argument("--frontier-size",type=int)
    run.add_argument("--window",type=int)
    sweep = sub.add_parser("sweep",help="Run controller component or sensitivity configurations")
    run_options(sweep)
    sweep.add_argument("--study",choices=("selection","components","allocation","width","capacity"),required=True)
    sweep.add_argument("--plan-only",action="store_true")
    sweep.add_argument("--reference-attempts",
                       help="Main LOCUS attempts reused for K=5 in the width study")
    frozen = sub.add_parser("repair-frozen",help="Repair from a fixed frontier and selected source condition")
    frozen.add_argument("--config",required=True)
    frozen_input = frozen.add_mutually_exclusive_group(required=True)
    frozen_input.add_argument("--task",help="One inference task as a JSON object")
    frozen_input.add_argument("--tasks",help="Inference tasks as JSONL")
    frozen.add_argument("--instance-id")
    frozen.add_argument("--condition",required=True)
    frozen.add_argument("--output",required=True)
    frozen.add_argument("--repeat",type=int,default=0)
    frozen.add_argument("--system",default="Frozen-edit")
    conditions = sub.add_parser("interventions",help="Construct frozen-edit source-selection conditions")
    conditions.add_argument("--packet",required=True,help="Packet file produced by --packet-dir")
    conditions.add_argument("--reference-support",help="JSON list of externally supplied support units")
    conditions.add_argument("--condition",choices=(
        "all","broad","selected","same-size-random","required-evidence-removed","duplicates-removed"),
        default="all")
    conditions.add_argument("--output",required=True)
    conditions.add_argument("--seed",type=int,default=0)
    evaluate = sub.add_parser("evaluate",help="Compute repair, evidence and resource metrics")
    evaluate.add_argument("--attempts",required=True,nargs="+")
    evaluate.add_argument("--grades")
    evaluate.add_argument("--annotations")
    evaluate.add_argument("--grading-metadata")
    evaluate.add_argument("--output",required=True)
    evaluate.add_argument("--graded-output")
    evaluate.add_argument("--locus-system",default="LOCUS")
    evaluate.add_argument("--expected-repeats",type=int,default=5)
    evaluate.add_argument("--bootstrap-samples",type=int,default=2000)
    evaluate.add_argument("--confidence",type=float,default=.95)
    evaluate.add_argument("--bootstrap-seed",type=int,default=0)
    evaluate.add_argument("--task-roster",nargs="+",help="Public inference task files defining complete rosters")
    export = sub.add_parser("export",help="Export one repeat to an official harness")
    export.add_argument("--attempts",required=True)
    export.add_argument("--output",required=True)
    export.add_argument("--repeat",type=int,required=True)
    export.add_argument("--format",choices=("swebench","pro-v1","harbor"),default="swebench")
    grade = sub.add_parser("grade-lite",help="Build or execute the official Lite grading command")
    grade.add_argument("--predictions",required=True)
    grade.add_argument("--run-id",required=True)
    grade.add_argument("--workers",type=int,default=1)
    grade.add_argument("--execute",action="store_true")
    pro1 = sub.add_parser("grade-pro-v1",help="Build or execute the official Pro V1 grading command")
    pro1.add_argument("--harness",required=True)
    pro1.add_argument("--grading-metadata",required=True)
    pro1.add_argument("--patches",required=True)
    pro1.add_argument("--output",required=True)
    pro1.add_argument("--workers",type=int,default=1)
    pro1.add_argument("--execute",action="store_true")
    pro2 = sub.add_parser("grade-pro-v2",help="Grade submitted patches in pristine official V2 images")
    pro2.add_argument("--attempts",required=True)
    pro2.add_argument("--tasks-dir",required=True)
    pro2.add_argument("--output",required=True)
    pro2.add_argument("--repeat",type=int)
    pro2.add_argument("--network",choices=("bridge","none"),default="bridge")
    read = sub.add_parser("read-grades",help="Read official harness reports into explicit grading records")
    read.add_argument("--format",choices=("lite","pro-v1"),required=True)
    read.add_argument("--input",required=True)
    read.add_argument("--repeat",type=int,required=True)
    read.add_argument("--output",required=True)
    merge = sub.add_parser("merge-grades")
    merge.add_argument("--attempts",required=True)
    merge.add_argument("--grades",required=True)
    merge.add_argument("--output",required=True)
    base = sub.add_parser("import-baseline",help="Import upstream patches and complete-task telemetry")
    base.add_argument("--predictions",required=True)
    base.add_argument("--tasks",required=True)
    base.add_argument("--system",required=True)
    base.add_argument("--model",required=True)
    base.add_argument("--repeat",type=int,required=True)
    base.add_argument("--telemetry")
    base.add_argument("--output",required=True)
    matrix = sub.add_parser("matrix",help="Write the 81 benchmark/backbone/system configurations")
    matrix.add_argument("--output",required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch":
            print(compact(fetch_public(args.benchmark,Path(args.output),revision=args.revision,limit=args.limit)))
        elif args.command == "run":
            settings,model = load_config(args.config)
            changes = {key:value for key,value in {
                "selection":args.selection,"allocation":args.allocation,"packet_tokens":args.packet_tokens,
                "frontier_size":args.frontier_size,"stability_window":args.window,
                "seed":args.seed}.items() if value is not None}
            _run_attempts(args,replace(settings,**changes).validate(),model)
        elif args.command == "sweep":
            from .experiments import cases
            settings,model = load_config(args.config)
            if args.seed is not None:
                settings = replace(settings, seed=args.seed).validate()
            all_cases = cases(args.study,settings)
            if args.plan_only:
                _json(Path(args.output)/"plan.json",[
                    dict(name=case.name,settings=asdict(case.settings)) for case in all_cases])
            else:
                shared = {}
                width_reference = _width_reference(args, settings, model) if args.study == "width" else None
                batch = args.batch_id
                for case in all_cases:
                    case_output = Path(args.output)/case.name/"attempts.jsonl"
                    manifest = _run_attempts(args,case.settings,model,
                                  output=case_output,system=case.name,reuse_from=shared,
                                  batch_id=batch, reference_settings=settings,
                                  width_reference=width_reference if case.name == "LOCUS-K5" else None)
                    batch = manifest["batch_id"]
                    if args.study == "allocation" and case.name == "LOCUS":
                        shared = {(row["instance_id"],row.get("repeat",0)):row
                                  for row in read_jsonl(case_output)}
        elif args.command == "repair-frozen":
            from .controller import run_agent
            if args.task:
                task = json.loads(Path(args.task).read_text(encoding="utf-8-sig"))
            else:
                selected = read_jsonl(args.tasks)
                if args.instance_id:
                    selected = [t for t in selected if t["instance_id"] == args.instance_id]
                if len(selected) != 1:
                    raise ValueError("Select one frozen inference task with --instance-id")
                task = selected[0]
            condition = json.loads(Path(args.condition).read_text(encoding="utf-8-sig"))
            settings,model = load_config(args.config)
            deadline = BENCHMARKS.get(task.get("benchmark"),{}).get("deadline",settings.deadline_seconds)
            settings = replace(settings,recovery=False,deadline_seconds=deadline,
                               packet_tokens=condition["token_cap"])
            record = run_agent(dict(task,repeat=args.repeat,system=args.system),settings,model,
                               frozen_packet=condition)
            write_jsonl(args.output,[record])
        elif args.command == "interventions":
            from .interventions import build_interventions,frozen_edit_intervention
            packet = json.loads(Path(args.packet).read_text(encoding="utf-8-sig"))
            support = json.loads(Path(args.reference_support).read_text(encoding="utf-8-sig")) \
                if args.reference_support else None
            if args.condition == "all":
                conditions = build_interventions(
                    packet["frontier"],packet.get("pool",packet["sources"]),packet["sources"],
                    reference_support_units=support,seed=args.seed,
                    requirements=packet.get("requirements"))
            else:
                conditions = {args.condition:frozen_edit_intervention(
                    args.condition,packet["frontier"],packet.get("pool",packet["sources"]),
                    packet["sources"],reference_support_units=support,seed=args.seed,
                    requirements=packet.get("requirements"))}
            for name,condition in conditions.items():
                _json(Path(args.output)/(name+".json"),condition)
        elif args.command == "evaluate":
            _evaluate(args)
        elif args.command == "export":
            rows = read_jsonl(args.attempts)
            if args.format == "harbor":
                from .benchmarks import export_harbor_replay
                export_harbor_replay(rows,Path(args.output),repeat=args.repeat)
            elif args.format == "pro-v1":
                export_pro_v1_patches(rows,Path(args.output),repeat=args.repeat)
            else:
                export_predictions(rows,Path(args.output),repeat=args.repeat)
        elif args.command == "grade-lite":
            _execute(lite_grade_command(Path(args.predictions),args.run_id,args.workers,sys.executable),args.execute)
        elif args.command == "grade-pro-v1":
            _execute(pro_v1_grade_command(Path(args.harness),Path(args.grading_metadata),Path(args.patches),
                                          Path(args.output),args.workers,sys.executable),args.execute,
                     cwd=Path(args.harness).resolve())
        elif args.command == "grade-pro-v2":
            rows = read_jsonl(args.attempts)
            if args.repeat is not None:
                rows = [r for r in rows if r.get("repeat",0) == args.repeat]
            grades = [grade_pro_v2(Path(args.tasks_dir)/r["instance_id"],r,network=args.network)
                      for r in rows]
            write_jsonl(args.output,grades)
        elif args.command == "read-grades":
            grades = read_lite_grades(Path(args.input),args.repeat) if args.format == "lite" \
                else read_pro_v1_grades(Path(args.input),args.repeat)
            write_jsonl(args.output,grades)
        elif args.command == "merge-grades":
            write_jsonl(args.output,merge_grades(read_jsonl(args.attempts),read_jsonl(args.grades)))
        elif args.command == "import-baseline":
            from .baselines import import_predictions,load_predictions
            telemetry = json.loads(Path(args.telemetry).read_text(encoding="utf-8-sig")) if args.telemetry else None
            write_jsonl(args.output,import_predictions(
                load_predictions(args.predictions),read_jsonl(args.tasks),system=args.system,
                model=args.model,repeat=args.repeat,telemetry=telemetry))
        elif args.command == "matrix":
            from .experiments import main_matrix
            _json(args.output,main_matrix())
        return 0
    except (ValueError,KeyError,TypeError,RuntimeError,OSError,subprocess.CalledProcessError) as exc:
        # No provider headers, credentials or response bodies are printed.
        print(f"{type(exc).__name__}: {str(exc)}",file=sys.stderr)
        return 2

