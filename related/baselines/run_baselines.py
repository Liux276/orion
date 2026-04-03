#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

# 统一模型标识 -> (yaml 节名, arch/name)
ARCH_MAP = {
    "RESNET50": ("resnet50", "resnet50"),
    "RESNET101": ("resnet101", "resnet101"),
    "MOBILENET_V2": ("mobilenet_v2", "mobilenet_v2"),
    "BERT": ("bert", "base"),
}

DEFAULT_MODEL_PAIRS = [
    ("RESNET50", "RESNET101"),
    ("RESNET50", "MOBILENET_V2"),
    ("MOBILENET_V2", "RESNET101"),
    ("BERT", "MOBILENET_V2"),
]

DEFAULT_POLICIES = ["Streams", "Isolated", "Sequential"]
DEFAULT_DISTRIBUTIONS = ["uniform", "poisson", "trace"]

MODEL_BATCH_SIZE = {
    "RESNET50": 32,
    "RESNET101": 32,
    "MOBILENET_V2": 32,
    "BERT": 4,
}

FIXED_LP_QPS = {
    "uniform": {
        "RESNET50": 20,
        "RESNET101": 15,
        "MOBILENET_V2": 60,
        "BERT": 3,
    },
    "poisson": {
        "RESNET50": 20,
        "RESNET101": 10,
        "MOBILENET_V2": 40,
        "BERT": 2,
    },
}

SLO_MS = {
    "RESNET50": 62.164785634333164,
    "RESNET101": 97.05058148032741,
    "MOBILENET_V2": 27.84495102731805,
    "BERT": 60.0538039694027,
}


def norm_dist(dist: str) -> str:
    dist = (dist or "").strip().lower()
    if dist in ("apollo", "appollo", "trace"):
        return "trace"
    return dist


def dist_dirname(dist: str) -> str:
    return "apollo" if norm_dist(dist) == "trace" else norm_dist(dist)


def parse_model_pair(spec: str):
    for sep in ("|", ":", ","):
        if sep in spec:
            left, right = [item.strip().upper() for item in spec.split(sep, 1)]
            if left not in ARCH_MAP or right not in ARCH_MAP:
                raise ValueError(f"Unknown model pair: {spec}")
            return left, right
    raise ValueError(f"Unsupported pair format: {spec}")


def load_trace_intervals(trace_path: Path):
    intervals = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"Trace file is empty or invalid: {trace_path}")
    return [float(item) for item in intervals]


def scale_trace_intervals(intervals, target_rps: int, num_requests: int):
    if target_rps <= 0:
        raise ValueError("target_rps must be positive for trace scaling")
    if num_requests > len(intervals):
        raise ValueError(
            f"apollo_num_requests={num_requests} exceeds trace length={len(intervals)}"
        )
    sliced = intervals[:num_requests]
    mean_interval = sum(sliced) / len(sliced)
    scale = (1.0 / target_rps) / mean_interval
    return [round(item * scale, 12) for item in sliced]


def stop_threshold_ms(model: str) -> float:
    return SLO_MS[model] * 2.0


def batch_size_for_model(model: str) -> int:
    return MODEL_BATCH_SIZE[model]


def fixed_lp_qps(model: str, distribution: str) -> int:
    distribution = norm_dist(distribution)
    if distribution == "trace":
        distribution = "poisson"
    try:
        return FIXED_LP_QPS[distribution][model]
    except KeyError as exc:
        raise ValueError(f"Missing fixed LP qps for model={model}, distribution={distribution}") from exc


def make_yaml(
    policy: str,
    hp_model: str,
    lp_model: str,
    hp_batch_size: int,
    lp_batch_size: int,
    hp_dist: str,
    hp_request_rate: int,
    lp_request_rate: int,
    hp_num_requests: int,
    lp_num_requests: int,
    seed: int,
    pin_memory: bool,
    trace_path: Path,
) -> str:
    hp_key, hp_arch = ARCH_MAP[hp_model]
    lp_key, lp_arch = ARCH_MAP[lp_model]
    distribution = norm_dist(hp_dist)

    return dedent(
        f"""\
        ---
        policy: "{policy}" # "MPS", "TickTock", "Streams", "Isolated", or "Sequential"
        models:
          model0:
            mode: eval
            name: {lp_key}
          model1:
            mode: eval
            name: {hp_key}

        shared_config:
          distribution: {distribution} # only controls HP
          trace_path: '{trace_path}'
          pin_memory: {"true" if pin_memory else "false"}
          seed: {seed}

        {lp_key}:
          arch: {lp_arch}
          batch_size: {lp_batch_size}
          num_iterations: {lp_num_requests}
          request_rate: {lp_request_rate}

        {hp_key}:
          arch: {hp_arch}
          batch_size: {hp_batch_size}
          num_iterations: {hp_num_requests}
          request_rate: {hp_request_rate}
        """
    )


def parse_metrics(json_path: Path):
    if not json_path.exists():
        return None, None, "missing metrics json"
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"json parse error: {exc}"

    def safe_float(key: str):
        try:
            return float(obj[key])
        except Exception:
            return None

    lp_metrics = {
        "p99_latency_ms": safe_float("p99-latency-0"),
        "throughput_rps": safe_float("throughput-0"),
    }
    hp_metrics = {
        "p99_latency_ms": safe_float("p99-latency-1"),
        "throughput_rps": safe_float("throughput-1"),
    }
    if None in hp_metrics.values() or None in lp_metrics.values():
        return None, None, "metrics json missing p99/throughput keys"
    return hp_metrics, lp_metrics, ""


def write_csv(rows, out_path: Path, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_row(
    *,
    hp_model,
    lp_model,
    hp_batch_size,
    lp_batch_size,
    distribution,
    policy,
    rps,
    lp_rps,
    gpu_id,
    hp_num_requests,
    lp_num_requests,
    run_dir,
    return_code,
    elapsed_sec,
    reused_existing,
    note,
    hp_metrics,
    lp_metrics,
):
    threshold = stop_threshold_ms(hp_model)
    hp_p99 = hp_metrics["p99_latency_ms"] if hp_metrics else ""
    hp_tpt = hp_metrics["throughput_rps"] if hp_metrics else ""
    lp_p99 = lp_metrics["p99_latency_ms"] if lp_metrics else ""
    lp_tpt = lp_metrics["throughput_rps"] if lp_metrics else ""
    exceeded = bool(hp_metrics and hp_metrics["p99_latency_ms"] > threshold)
    return {
        "models": f"{hp_model}|{lp_model}",
        "batch_size": f"{hp_batch_size}|{lp_batch_size}",
        "distribution": dist_dirname(distribution),
        "policy": policy,
        "rps": rps,
        "hp_rps": rps,
        "lp_rps": lp_rps,
        "gpu_id": gpu_id,
        "hp_num_requests": hp_num_requests,
        "lp_num_requests": lp_num_requests,
        "hp_stop_threshold_ms": round(threshold, 6),
        "hp_p99_ms": hp_p99,
        "hp_throughput_rps": hp_tpt,
        "lp_p99_ms": lp_p99,
        "lp_throughput_rps": lp_tpt,
        "exceeded_stop_threshold": exceeded,
        "return_code": return_code,
        "elapsed_sec": elapsed_sec,
        "reused_existing": reused_existing,
        "run_dir": str(run_dir),
        "note": note,
    }


def launch_policy_runs(run_specs, workdir: Path, python_exec: str, timeout: int):
    active = []
    for spec in run_specs:
        stdout_path = spec["run_dir"] / "run.stdout.log"
        stderr_path = spec["run_dir"] / "run.stderr.log"
        stdout_handle = open(stdout_path, "w", encoding="utf-8")
        stderr_handle = open(stderr_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(spec["gpu_id"])
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            python_exec,
            "main.py",
            "--config",
            str(spec["config_path"]),
            "--log",
            str(spec["log_path"]),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
        )
        active.append(
            {
                "spec": spec,
                "proc": proc,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
                "started_at": time.time(),
            }
        )

    finished = []
    for item in active:
        proc = item["proc"]
        note = ""
        code = 0
        try:
            if timeout > 0:
                proc.wait(timeout=timeout)
            else:
                proc.wait()
            code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            code = -9
            note = f"timeout({timeout}s)"
        finally:
            item["stdout_handle"].close()
            item["stderr_handle"].close()

        item["return_code"] = code
        item["elapsed_sec"] = round(time.time() - item["started_at"], 3)
        item["note"] = note
        finished.append(item)
    return finished


def main():
    parser = argparse.ArgumentParser(
        description="Sweep HP/LP request_rate from rps=1, run baselines on isolated GPUs, and stop when HP p99 > 2x SLO."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run main.py")
    parser.add_argument(
        "--workdir",
        default=str(Path(__file__).resolve().parent),
        help="Working directory containing main.py",
    )
    parser.add_argument(
        "--model-pairs",
        nargs="+",
        default=[f"{hp}|{lp}" for hp, lp in DEFAULT_MODEL_PAIRS],
        help="Model pairs to run, e.g. RESNET50|RESNET101 BERT|MOBILENET_V2",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        choices=DEFAULT_POLICIES,
        help="Scheduling policies to run in parallel on different GPUs",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=DEFAULT_DISTRIBUTIONS,
        help="HP distributions to evaluate: uniform, poisson, apollo/trace",
    )
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="GPU ids assigned to policies in order",
    )
    parser.add_argument("--start-rps", type=int, default=1, help="Initial HP request_rate")
    parser.add_argument("--rps-step", type=int, default=1, help="Increment applied after each sweep step")
    parser.add_argument("--max-rps", type=int, default=512, help="Safety cap for the request_rate sweep")
    parser.add_argument("--num-requests", type=int, default=500, help="num_iterations for non-apollo runs")
    parser.add_argument(
        "--apollo-num-requests",
        type=int,
        default=6240,
        help="num_iterations for apollo(trace) runs; must not exceed trace length",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed written to config.yaml")
    parser.add_argument("--pin-memory", action="store_true", default=True, help="Use pin_memory=true (default)")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Use pin_memory=false")
    parser.add_argument(
        "--trace-path",
        default="./inter_arrival_times.json",
        help="Base trace file used for apollo(trace) runs",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Per-run timeout in seconds (0 means no timeout)")
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs; skip launching main.py")
    parser.add_argument("--force", action="store_true", help="Re-run even if metrics json already exists")
    args = parser.parse_args()

    if len(args.gpu_ids) < len(args.policies):
        raise ValueError(
            f"Need at least {len(args.policies)} gpu ids for policies {args.policies}, got {args.gpu_ids}"
        )
    if args.start_rps <= 0 or args.rps_step <= 0 or args.max_rps <= 0:
        raise ValueError("start-rps, rps-step and max-rps must be positive")
    if args.num_requests <= 0 or args.apollo_num_requests <= 0:
        raise ValueError("num-requests and apollo-num-requests must be positive")

    workdir = Path(args.workdir).resolve()
    trace_path = Path(args.trace_path)
    if not trace_path.is_absolute():
        trace_path = (workdir / trace_path).resolve()
    trace_intervals = load_trace_intervals(trace_path)
    if args.apollo_num_requests > len(trace_intervals):
        raise ValueError(
            f"--apollo-num-requests={args.apollo_num_requests} exceeds trace length={len(trace_intervals)}"
        )

    model_pairs = [parse_model_pair(item) for item in args.model_pairs]
    distributions = [norm_dist(item) for item in args.distributions]
    gpu_map = {policy: gpu_id for policy, gpu_id in zip(args.policies, args.gpu_ids)}

    result_root = workdir / "result"
    result_root.mkdir(parents=True, exist_ok=True)

    run_rows = []
    stop_rows = []
    fieldnames = [
        "models",
        "batch_size",
        "distribution",
        "policy",
        "rps",
        "hp_rps",
        "lp_rps",
        "gpu_id",
        "hp_num_requests",
        "lp_num_requests",
        "hp_stop_threshold_ms",
        "hp_p99_ms",
        "hp_throughput_rps",
        "lp_p99_ms",
        "lp_throughput_rps",
        "exceeded_stop_threshold",
        "return_code",
        "elapsed_sec",
        "reused_existing",
        "run_dir",
        "note",
    ]

    print(f"[INFO] workdir={workdir}")
    print(f"[INFO] policy->gpu mapping: {gpu_map}")

    overall_t0 = time.time()
    for hp_model, lp_model in model_pairs:
        hp_batch_size = batch_size_for_model(hp_model)
        lp_batch_size = batch_size_for_model(lp_model)
        for distribution in distributions:
            lp_request_rate = fixed_lp_qps(lp_model, distribution)
            active_policies = set(args.policies)
            rps = args.start_rps
            while active_policies and rps <= args.max_rps:
                print(
                    f"\n[SWEEP] models={hp_model}|{lp_model} batch={hp_batch_size}|{lp_batch_size} "
                    f"distribution={dist_dirname(distribution)} hp_rps={rps} lp_rps={lp_request_rate} "
                    f"active={sorted(active_policies)}"
                )
                num_requests = args.apollo_num_requests if norm_dist(distribution) == "trace" else args.num_requests
                run_specs = []

                for policy in args.policies:
                    if policy not in active_policies:
                        continue

                    hp_tag = ARCH_MAP[hp_model][0]
                    lp_tag = ARCH_MAP[lp_model][0]
                    run_dir = (
                        result_root
                        / f"{hp_tag}_{lp_tag}"
                        / policy
                        / dist_dirname(distribution)
                        / f"batch_{hp_batch_size}_{lp_batch_size}"
                        / f"rps_{rps}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    if norm_dist(distribution) == "trace":
                        scaled_trace_path = run_dir / f"trace_rps_{rps}.json"
                        scaled_trace = scale_trace_intervals(trace_intervals, rps, num_requests)
                        scaled_trace_path.write_text(
                            json.dumps(scaled_trace, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        run_trace_path = scaled_trace_path.resolve()
                    else:
                        run_trace_path = trace_path.resolve()

                    config_path = run_dir / "config.yaml"
                    config_text = make_yaml(
                        policy=policy,
                        hp_model=hp_model,
                        lp_model=lp_model,
                        hp_batch_size=hp_batch_size,
                        lp_batch_size=lp_batch_size,
                        hp_dist=distribution,
                        hp_request_rate=rps,
                        lp_request_rate=lp_request_rate,
                        hp_num_requests=num_requests,
                        lp_num_requests=num_requests,
                        seed=args.seed,
                        pin_memory=args.pin_memory,
                        trace_path=run_trace_path,
                    )
                    config_path.write_text(config_text, encoding="utf-8")

                    log_name = f"eval-{lp_tag}eval-{hp_tag}.log"
                    log_path = run_dir / log_name
                    metrics_path = run_dir / f"{log_name}.json"

                    run_specs.append(
                        {
                            "policy": policy,
                            "distribution": distribution,
                            "hp_model": hp_model,
                            "lp_model": lp_model,
                            "hp_batch_size": hp_batch_size,
                            "lp_batch_size": lp_batch_size,
                            "rps": rps,
                            "lp_rps": lp_request_rate,
                            "gpu_id": gpu_map[policy],
                            "hp_num_requests": num_requests,
                            "lp_num_requests": num_requests,
                            "run_dir": run_dir,
                            "config_path": config_path,
                            "log_path": log_path,
                            "metrics_path": metrics_path,
                        }
                    )

                if not run_specs:
                    break

                launched = []
                for spec in run_specs:
                    hp_metrics = lp_metrics = None
                    note = ""
                    reused_existing = False
                    if not args.force and spec["metrics_path"].exists():
                        hp_metrics, lp_metrics, note = parse_metrics(spec["metrics_path"])
                        if hp_metrics and lp_metrics:
                            reused_existing = True
                            launched.append(
                                {
                                    "spec": spec,
                                    "return_code": 0,
                                    "elapsed_sec": 0.0,
                                    "note": note,
                                    "hp_metrics": hp_metrics,
                                    "lp_metrics": lp_metrics,
                                    "reused_existing": reused_existing,
                                }
                            )
                            continue

                    if args.dry_run:
                        launched.append(
                            {
                                "spec": spec,
                                "return_code": 0,
                                "elapsed_sec": 0.0,
                                "note": "dry-run",
                                "hp_metrics": None,
                                "lp_metrics": None,
                                "reused_existing": False,
                            }
                        )

                missing_specs = [
                    spec
                    for spec in run_specs
                    if not any(item["spec"] is spec for item in launched)
                ]
                if missing_specs and not args.dry_run:
                    finished = launch_policy_runs(
                        missing_specs,
                        workdir=workdir,
                        python_exec=args.python,
                        timeout=args.timeout,
                    )
                    for item in finished:
                        hp_metrics, lp_metrics, parse_note = parse_metrics(item["spec"]["metrics_path"])
                        note = item["note"]
                        if parse_note:
                            note = f"{note} | {parse_note}".strip(" |")
                        launched.append(
                            {
                                "spec": item["spec"],
                                "return_code": item["return_code"],
                                "elapsed_sec": item["elapsed_sec"],
                                "note": note,
                                "hp_metrics": hp_metrics,
                                "lp_metrics": lp_metrics,
                                "reused_existing": False,
                            }
                        )

                launched.sort(key=lambda item: args.policies.index(item["spec"]["policy"]))

                for item in launched:
                    spec = item["spec"]
                    row = build_run_row(
                        hp_model=spec["hp_model"],
                        lp_model=spec["lp_model"],
                        hp_batch_size=spec["hp_batch_size"],
                        lp_batch_size=spec["lp_batch_size"],
                        distribution=spec["distribution"],
                        policy=spec["policy"],
                        rps=spec["rps"],
                        lp_rps=spec["lp_rps"],
                        gpu_id=spec["gpu_id"],
                        hp_num_requests=spec["hp_num_requests"],
                        lp_num_requests=spec["lp_num_requests"],
                        run_dir=spec["run_dir"],
                        return_code=item["return_code"],
                        elapsed_sec=item["elapsed_sec"],
                        reused_existing=item["reused_existing"],
                        note=item["note"],
                        hp_metrics=item["hp_metrics"],
                        lp_metrics=item["lp_metrics"],
                    )
                    run_rows.append(row)

                    should_stop = False
                    stop_reason = ""
                    if args.dry_run:
                        should_stop = True
                        stop_reason = "dry_run"
                    elif item["return_code"] != 0:
                        should_stop = True
                        stop_reason = "run_error"
                    elif not item["hp_metrics"] or not item["lp_metrics"]:
                        should_stop = True
                        stop_reason = "metrics_missing"
                    elif item["hp_metrics"]["p99_latency_ms"] > stop_threshold_ms(spec["hp_model"]):
                        should_stop = True
                        stop_reason = "hp_p99_gt_2xslo"

                    print(
                        f"[DONE] policy={spec['policy']} gpu={spec['gpu_id']} hp_rps={spec['rps']} lp_rps={spec['lp_rps']} "
                        f"hp_p99={row['hp_p99_ms']} hp_tpt={row['hp_throughput_rps']} "
                        f"lp_p99={row['lp_p99_ms']} lp_tpt={row['lp_throughput_rps']} "
                        f"stop={should_stop} reason={stop_reason or '-'}"
                    )

                    if should_stop and spec["policy"] in active_policies:
                        active_policies.remove(spec["policy"])
                        stop_rows.append({**row, "stop_reason": stop_reason})

                if args.dry_run:
                    break
                rps += args.rps_step

            for policy in args.policies:
                if policy in active_policies:
                    stop_rows.append(
                        {
                            "models": f"{hp_model}|{lp_model}",
                            "batch_size": f"{hp_batch_size}|{lp_batch_size}",
                            "distribution": dist_dirname(distribution),
                            "policy": policy,
                            "rps": "",
                            "hp_rps": "",
                            "lp_rps": lp_request_rate,
                            "gpu_id": gpu_map[policy],
                            "hp_num_requests": "",
                            "lp_num_requests": "",
                            "hp_stop_threshold_ms": round(stop_threshold_ms(hp_model), 6),
                            "hp_p99_ms": "",
                            "hp_throughput_rps": "",
                            "lp_p99_ms": "",
                            "lp_throughput_rps": "",
                            "exceeded_stop_threshold": False,
                            "return_code": "",
                            "elapsed_sec": "",
                            "reused_existing": False,
                            "run_dir": "",
                            "note": f"max_rps_reached({args.max_rps})",
                            "stop_reason": "max_rps_reached",
                        }
                    )

    write_csv(run_rows, result_root / "all_runs.csv", fieldnames)
    write_csv(stop_rows, result_root / "threshold_summary.csv", fieldnames + ["stop_reason"])

    print(f"\n[INFO] wrote {result_root / 'all_runs.csv'}")
    print(f"[INFO] wrote {result_root / 'threshold_summary.csv'}")
    print(f"[INFO] finished in {round(time.time() - overall_t0, 2)}s")


if __name__ == "__main__":
    main()
