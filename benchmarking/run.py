#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ARCH_TO_TAG = {
    "RESNET50": "resnet50",
    "RESNET101": "resnet101",
    "MOBILENET_V2": "mobilenet_v2",
    "BERT": "bert",
}

MODEL_BATCH_SIZE = {
    "RESNET50": 32,
    "RESNET101": 32,
    "MOBILENET_V2": 32,
    "BERT": 4,
}

NUM_KERNELS = {
    "resnet50": 175,
    "resnet101": 345,
    "mobilenet_v2": 152,
    "bert": 601,
}

UNIFORM_LP_QPS = {
    "RESNET50": {4: 60, 32: 20},
    "RESNET101": {4: 30, 32: 15},
    "MOBILENET_V2": {4: 80, 32: 60},
    "BERT": {4: 6, 32: 2},
}

# 这里与 related/baselines/run_baselines.py 保持一致：
# 前三个视觉模型的基础 SLO 已经按你的说明乘过 2，真正停止阈值仍然是 2x SLO。
SLO_MS = {
    "RESNET50": 31.082392817166582 * 2,
    "RESNET101": 48.525290740163705 * 2,
    "MOBILENET_V2": 13.922475513659025 * 2,
    "BERT": 60.0538039694027,
}

DEFAULT_MODEL_PAIRS = [
    ("RESNET50", "RESNET101"),
    ("RESNET50", "MOBILENET_V2"),
    ("MOBILENET_V2", "RESNET101"),
    ("BERT", "MOBILENET_V2"),
]

DEFAULT_DISTRIBUTIONS = ["uniform", "poisson", "apollo"]
DEFAULT_STRATEGIES = ["Orion", "REEF"]


def normalize_distribution(label: str) -> str:
    label = (label or "").strip().lower()
    if label in {"trace", "apollo", "appollo"}:
        return "apollo"
    if label not in {"uniform", "poisson"}:
        raise ValueError(f"Unsupported distribution: {label}")
    return label


def parse_model_pair(spec: str):
    for sep in ("|", ":", ","):
        if sep in spec:
            hp, lp = [item.strip().upper() for item in spec.split(sep, 1)]
            if hp not in ARCH_TO_TAG or lp not in ARCH_TO_TAG:
                raise ValueError(f"Unknown model pair: {spec}")
            return hp, lp
    raise ValueError(f"Unsupported model pair format: {spec}")


def stop_threshold_ms(model_key: str) -> float:
    return SLO_MS[model_key] * 2.0


def batch_size_for_model(model_key: str) -> int:
    return MODEL_BATCH_SIZE[model_key]


def lp_qps_for_model(model_key: str) -> int:
    batch_size = batch_size_for_model(model_key)
    return UNIFORM_LP_QPS[model_key][batch_size]


def load_trace_intervals(trace_path: Path):
    intervals = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"Trace file is empty or invalid: {trace_path}")
    return [float(item) for item in intervals]


def scale_trace_intervals(intervals, target_rps: int, num_requests: int):
    if target_rps <= 0:
        raise ValueError("target_rps must be positive for apollo runs")
    if num_requests > len(intervals):
        raise ValueError(
            f"apollo_num_requests={num_requests} exceeds trace length={len(intervals)}"
        )
    sliced = intervals[:num_requests]
    mean_interval = sum(sliced) / len(sliced)
    scale = (1.0 / target_rps) / mean_interval
    return [round(item * scale, 12) for item in sliced]


def build_entry(
    model_key: str,
    batch_size: int,
    request_rate: int,
    num_iters: int,
    is_hp: bool,
    hp_distribution: str,
    kernel_root: Path,
    trace_input: Optional[Path],
):
    model_tag = ARCH_TO_TAG[model_key]
    args = {
        "model_name": model_tag,
        "batchsize": batch_size,
        "rps": request_rate,
        "uniform": True if not is_hp else hp_distribution == "uniform",
        "dummy_data": True,
        "train": False,
    }
    if is_hp and hp_distribution == "apollo" and trace_input is not None:
        args["input_file"] = str(trace_input)

    return {
        "arch": model_tag,
        "kernel_file": str(kernel_root / f"{model_tag}_{batch_size}_fwd"),
        "num_kernels": NUM_KERNELS[model_tag],
        "num_iters": num_iters,
        "args": args,
    }


def build_config_json(
    hp_key: str,
    lp_key: str,
    hp_rps: int,
    lp_rps: int,
    hp_distribution: str,
    num_requests: int,
    kernel_root: Path,
    trace_input: Optional[Path],
):
    hp_batch_size = batch_size_for_model(hp_key)
    lp_batch_size = batch_size_for_model(lp_key)
    lp_entry = build_entry(
        model_key=lp_key,
        batch_size=lp_batch_size,
        request_rate=lp_rps,
        num_iters=num_requests,
        is_hp=False,
        hp_distribution=hp_distribution,
        kernel_root=kernel_root,
        trace_input=None,
    )
    hp_entry = build_entry(
        model_key=hp_key,
        batch_size=hp_batch_size,
        request_rate=hp_rps,
        num_iters=num_requests,
        is_hp=True,
        hp_distribution=hp_distribution,
        kernel_root=kernel_root,
        trace_input=trace_input,
    )
    # launch_jobs.py 里 client 0 视为 LP，client 1 视为 HP，顺序必须保持 [LP, HP]。
    return [lp_entry, hp_entry]


def json_cell(metrics):
    if not metrics:
        return ""
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_metrics_json(path: Path):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metrics = {
        "p50_latency": safe_float(payload.get("p50_latency")),
        "p95_latency": safe_float(payload.get("p95_latency")),
        "p99_latency": safe_float(payload.get("p99_latency")),
        "throughput": safe_float(payload.get("throughput")),
    }
    if metrics["throughput"] is None:
        return None
    if (
        metrics["p50_latency"] is None
        or metrics["p95_latency"] is None
        or metrics["p99_latency"] is None
    ):
        return {"throughput": metrics["throughput"]}
    return metrics


def remove_stale_metric_files(repo_root: Path):
    for name in ("client_0.json", "client_1.json", "be.json", "hp.json"):
        path = repo_root / name
        if path.exists():
            path.unlink()


def build_summary(
    *,
    strategy: str,
    hp_key: str,
    lp_key: str,
    distribution: str,
    hp_rps: int,
    lp_rps: int,
    num_requests: int,
    gpu_id: int,
    run_dir: Path,
    return_code,
    elapsed_sec,
    reused_existing,
    note,
    hp_metrics,
    lp_metrics,
):
    hp_batch_size = batch_size_for_model(hp_key)
    lp_batch_size = batch_size_for_model(lp_key)
    threshold = stop_threshold_ms(hp_key)
    hp_p99 = safe_float((hp_metrics or {}).get("p99_latency"))
    hp_tpt = safe_float((hp_metrics or {}).get("throughput"))
    lp_p99 = safe_float((lp_metrics or {}).get("p99_latency"))
    lp_tpt = safe_float((lp_metrics or {}).get("throughput"))

    return {
        "strategy": strategy,
        "models": f"{hp_key}|{lp_key}",
        "bs": f"{hp_batch_size}|{lp_batch_size}",
        "distribution": distribution,
        "rps": hp_rps,
        "hp_rps": hp_rps,
        "lp_rps": lp_rps,
        "gpu_id": gpu_id,
        "hp_num_requests": num_requests,
        "lp_num_requests": num_requests,
        "hp_stop_threshold_ms": round(threshold, 6),
        "hp_p99_ms": hp_p99 if hp_p99 is not None else "",
        "hp_throughput_rps": hp_tpt if hp_tpt is not None else "",
        "lp_p99_ms": lp_p99 if lp_p99 is not None else "",
        "lp_throughput_rps": lp_tpt if lp_tpt is not None else "",
        "exceeded_stop_threshold": bool(hp_p99 is not None and hp_p99 > threshold),
        "return_code": return_code,
        "elapsed_sec": elapsed_sec,
        "reused_existing": reused_existing,
        "run_dir": str(run_dir.resolve()),
        "note": note,
        "hp": hp_metrics or {},
        "lp": lp_metrics or {},
    }


def summary_fieldnames():
    return [
        "strategy",
        "models",
        "bs",
        "distribution",
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
        "hp",
        "lp",
    ]


def summary_to_row(summary):
    row = {key: summary.get(key, "") for key in summary_fieldnames()}
    row["hp"] = json_cell(summary.get("hp"))
    row["lp"] = json_cell(summary.get("lp"))
    return row


def write_csv(rows, out_path: Path, extra_fields=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = summary_fieldnames()
    if extra_fields:
        fieldnames = fieldnames + list(extra_fields)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(record)


def load_existing_summary(summary_path: Path):
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if "strategy" not in summary or "models" not in summary:
        return None
    summary["reused_existing"] = True
    return summary


def execute_run(
    *,
    repo_root: Path,
    python_exe: str,
    launch_py: Path,
    orion_preload: str,
    timeout: int,
    gpu_id: int,
    strategy: str,
    config_path: Path,
    run_dir: Path,
    hp_key: str,
    lp_key: str,
    distribution: str,
    hp_rps: int,
    lp_rps: int,
    num_requests: int,
    force: bool,
):
    summary_path = run_dir / "summary.json"
    if not force:
        existing = load_existing_summary(summary_path)
        if existing is not None:
            return existing

    stdout_path = run_dir / "run.stdout.log"
    stderr_path = run_dir / "run.stderr.log"
    cmd_path = run_dir / "cmd.txt"
    elapsed_path = run_dir / "elapsed.txt"

    cmd = [
        python_exe,
        str(launch_py),
        "--algo",
        strategy.lower(),
        "--config_file",
        str(config_path),
    ]
    remove_stale_metric_files(repo_root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    env["ORION_ROOT"] = str(repo_root)
    env["LD_PRELOAD"] = os.path.expandvars(orion_preload)

    cmd_record = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"LD_PRELOAD={env['LD_PRELOAD']} "
        f"{shlex.join(cmd)}"
    )
    cmd_path.write_text(cmd_record + "\n", encoding="utf-8")

    return_code = 0
    note = ""
    started_at = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            try:
                if timeout > 0:
                    proc.wait(timeout=timeout)
                else:
                    proc.wait()
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                return_code = -9
                note = f"timeout({timeout}s)"
    except Exception as exc:
        return_code = -1
        note = str(exc)

    elapsed_sec = round(time.time() - started_at, 3)
    elapsed_path.write_text(f"{elapsed_sec}\n", encoding="utf-8")

    client0_src = repo_root / "client_0.json"
    client1_src = repo_root / "client_1.json"
    client0_dst = run_dir / "client_0.json"
    client1_dst = run_dir / "client_1.json"
    lp_metrics = hp_metrics = None

    if client0_src.exists():
        shutil.copy2(client0_src, client0_dst)
        lp_metrics = load_metrics_json(client0_dst)
    if client1_src.exists():
        shutil.copy2(client1_src, client1_dst)
        hp_metrics = load_metrics_json(client1_dst)

    summary = build_summary(
        strategy=strategy,
        hp_key=hp_key,
        lp_key=lp_key,
        distribution=distribution,
        hp_rps=hp_rps,
        lp_rps=lp_rps,
        num_requests=num_requests,
        gpu_id=gpu_id,
        run_dir=run_dir,
        return_code=return_code,
        elapsed_sec=elapsed_sec,
        reused_existing=False,
        note=note,
        hp_metrics=hp_metrics,
        lp_metrics=lp_metrics,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_run_dir(result_root: Path, strategy: str, hp_key: str, lp_key: str, distribution: str, rps: int):
    hp_batch_size = batch_size_for_model(hp_key)
    lp_batch_size = batch_size_for_model(lp_key)
    hp_tag = ARCH_TO_TAG[hp_key]
    lp_tag = ARCH_TO_TAG[lp_key]
    return (
        result_root
        / strategy.lower()
        / f"{hp_tag}_{lp_tag}"
        / distribution
        / f"batch_{hp_batch_size}_{lp_batch_size}"
        / f"rps_{rps}"
    )


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root_default = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Sweep HP request_rate from rps=1 for Orion/REEF, stop when HP p99 exceeds 2x SLO, "
            "and store per-rps summaries for later collection."
        )
    )
    parser.add_argument("--repo-root", default=str(repo_root_default), help="仓库根目录")
    parser.add_argument(
        "--result-root",
        default=str(script_dir / "benchmarking" / "result"),
        help="结果根目录",
    )
    parser.add_argument(
        "--kernel-root",
        default=str(script_dir / "model_kernels" / "a10"),
        help="内核文件根目录",
    )
    parser.add_argument(
        "--trace-input",
        default=str(repo_root_default / "artifact_evaluation" / "fig10" / "inter_arrival_times.json"),
        help="apollo(trace) 的基础到达间隔文件",
    )
    parser.add_argument(
        "--model-pairs",
        nargs="+",
        default=[f"{hp}|{lp}" for hp, lp in DEFAULT_MODEL_PAIRS],
        help="要跑的模型组合，格式如 RESNET50|RESNET101",
    )
    parser.add_argument(
        "--distributions",
        default="uniform,poisson,apollo",
        help="HP 分布，逗号分隔；支持 uniform, poisson, apollo",
    )
    parser.add_argument(
        "--strategies",
        default="Orion,REEF",
        help="策略顺序，逗号分隔；默认严格顺序运行 Orion,REEF",
    )
    parser.add_argument("--start-rps", type=int, default=1, help="初始 HP request_rate")
    parser.add_argument("--rps-step", type=int, default=1, help="每轮递增的 HP request_rate")
    parser.add_argument("--max-rps", type=int, default=512, help="安全上限，避免无限 sweep")
    parser.add_argument("--num-requests", type=int, default=500, help="非 apollo 的请求数")
    parser.add_argument("--apollo-num-requests", type=int, default=6240, help="apollo 的请求数")
    parser.add_argument("--gpu-id", type=int, default=3, help="绑定测试的物理 GPU id")
    parser.add_argument("--python", default=sys.executable, help="用于运行 launch_jobs.py 的解释器")
    parser.add_argument(
        "--orion-preload",
        default=str(repo_root_default / "src" / "cuda_capture" / "libinttemp.so"),
        help="运行所需的 LD_PRELOAD",
    )
    parser.add_argument("--timeout", type=int, default=0, help="单次运行超时（秒），0 表示不限时")
    parser.add_argument("--force", action="store_true", help="即使已有 summary.json 也强制重跑")
    parser.add_argument(
        "--gen-only",
        action="store_true",
        help="仅生成 rps=start-rps 的配置目录和 config，不实际执行",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="兼容旧接口的占位参数；当前脚本默认执行 sweep，除非显式传 --gen-only",
    )
    args = parser.parse_args()

    if args.start_rps <= 0 or args.rps_step <= 0 or args.max_rps <= 0:
        raise ValueError("start-rps、rps-step 和 max-rps 必须是正整数")
    if args.num_requests <= 0 or args.apollo_num_requests <= 0:
        raise ValueError("num-requests 和 apollo-num-requests 必须是正整数")

    repo_root = Path(args.repo_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    kernel_root = Path(args.kernel_root).expanduser().resolve()
    trace_input = Path(args.trace_input).expanduser().resolve()
    launch_py = repo_root / "benchmarking" / "launch_jobs.py"
    model_pairs = [parse_model_pair(spec) for spec in args.model_pairs]
    distributions = [normalize_distribution(item) for item in args.distributions.split(",") if item.strip()]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    trace_intervals = load_trace_intervals(trace_input) if "apollo" in distributions else []

    if DEFAULT_STRATEGIES != strategies[: len(DEFAULT_STRATEGIES)]:
        print(f"[WARN] current strategy order: {strategies}", flush=True)
    print(f"[INFO] repo_root={repo_root}")
    print(f"[INFO] result_root={result_root}")
    print(f"[INFO] gpu_id={args.gpu_id}")
    print(f"[INFO] strategies={strategies}")

    result_root.mkdir(parents=True, exist_ok=True)
    run_rows = []
    stop_rows = []

    for hp_key, lp_key in model_pairs:
        lp_rps = lp_qps_for_model(lp_key)
        hp_threshold = stop_threshold_ms(hp_key)
        for distribution in distributions:
            active_strategies = list(strategies)
            rps = args.start_rps

            while active_strategies and rps <= args.max_rps:
                num_requests = args.apollo_num_requests if distribution == "apollo" else args.num_requests
                print(
                    f"[SWEEP] models={hp_key}|{lp_key} bs={batch_size_for_model(hp_key)}|{batch_size_for_model(lp_key)} "
                    f"distribution={distribution} hp_rps={rps} lp_rps={lp_rps} "
                    f"threshold_ms={round(hp_threshold, 6)} active={active_strategies}",
                    flush=True,
                )

                for strategy in list(active_strategies):
                    run_dir = make_run_dir(result_root, strategy, hp_key, lp_key, distribution, rps)
                    run_dir.mkdir(parents=True, exist_ok=True)

                    trace_path = None
                    if distribution == "apollo":
                        scaled_trace = scale_trace_intervals(trace_intervals, rps, num_requests)
                        trace_path = run_dir / f"trace_rps_{rps}.json"
                        trace_path.write_text(
                            json.dumps(scaled_trace, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                    config = build_config_json(
                        hp_key=hp_key,
                        lp_key=lp_key,
                        hp_rps=rps,
                        lp_rps=lp_rps,
                        hp_distribution=distribution,
                        num_requests=num_requests,
                        kernel_root=kernel_root,
                        trace_input=trace_path,
                    )
                    config_path = run_dir / "config.json"
                    config_path.write_text(
                        json.dumps(config, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

                    if args.gen_only:
                        summary = build_summary(
                            strategy=strategy,
                            hp_key=hp_key,
                            lp_key=lp_key,
                            distribution=distribution,
                            hp_rps=rps,
                            lp_rps=lp_rps,
                            num_requests=num_requests,
                            gpu_id=args.gpu_id,
                            run_dir=run_dir,
                            return_code="",
                            elapsed_sec="",
                            reused_existing=False,
                            note="gen-only",
                            hp_metrics=None,
                            lp_metrics=None,
                        )
                        (run_dir / "summary.json").write_text(
                            json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    else:
                        summary = execute_run(
                            repo_root=repo_root,
                            python_exe=args.python,
                            launch_py=launch_py,
                            orion_preload=args.orion_preload,
                            timeout=args.timeout,
                            gpu_id=args.gpu_id,
                            strategy=strategy,
                            config_path=config_path,
                            run_dir=run_dir,
                            hp_key=hp_key,
                            lp_key=lp_key,
                            distribution=distribution,
                            hp_rps=rps,
                            lp_rps=lp_rps,
                            num_requests=num_requests,
                            force=args.force,
                        )

                    run_rows.append(summary_to_row(summary))

                    stop_reason = ""
                    should_stop = False
                    if args.gen_only:
                        should_stop = True
                        stop_reason = "gen_only"
                    elif summary.get("return_code") not in (0, "0"):
                        should_stop = True
                        stop_reason = "run_error"
                    elif safe_float(summary.get("hp_p99_ms")) is None:
                        should_stop = True
                        stop_reason = "metrics_missing"
                    elif bool(summary.get("exceeded_stop_threshold")):
                        should_stop = True
                        stop_reason = "hp_p99_gt_2xslo"

                    print(
                        f"[DONE] strategy={strategy} distribution={distribution} hp_rps={rps} "
                        f"hp_p99_ms={summary.get('hp_p99_ms', '')} hp_tpt={summary.get('hp_throughput_rps', '')} "
                        f"lp_p99_ms={summary.get('lp_p99_ms', '')} lp_tpt={summary.get('lp_throughput_rps', '')} "
                        f"stop={should_stop} reason={stop_reason or '-'}",
                        flush=True,
                    )

                    if should_stop:
                        stop_rows.append({**summary_to_row(summary), "stop_reason": stop_reason})
                        active_strategies.remove(strategy)

                if args.gen_only:
                    break
                rps += args.rps_step

            for strategy in active_strategies:
                stop_rows.append(
                    {
                        "strategy": strategy,
                        "models": f"{hp_key}|{lp_key}",
                        "bs": f"{batch_size_for_model(hp_key)}|{batch_size_for_model(lp_key)}",
                        "distribution": distribution,
                        "rps": args.max_rps,
                        "hp_rps": args.max_rps,
                        "lp_rps": lp_rps,
                        "gpu_id": args.gpu_id,
                        "hp_num_requests": "",
                        "lp_num_requests": "",
                        "hp_stop_threshold_ms": round(hp_threshold, 6),
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
                        "hp": "",
                        "lp": "",
                        "stop_reason": "max_rps_reached",
                    }
                )

    all_runs_csv = result_root / "all_runs.csv"
    threshold_csv = result_root / "threshold_summary.csv"
    write_csv(run_rows, all_runs_csv)
    write_csv(stop_rows, threshold_csv, extra_fields=["stop_reason"])
    print(f"[INFO] wrote {all_runs_csv}")
    print(f"[INFO] wrote {threshold_csv}")


if __name__ == "__main__":
    main()
