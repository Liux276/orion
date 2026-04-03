#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

ARCH_MAP = {
    "RESNET50": ("resnet50", "resnet50"),
    "RESNET101": ("resnet101", "resnet101"),
    "MOBILENET_V2": ("mobilenet_v2", "mobilenet_v2"),
    "BERT": ("bert", "base"),
}
TAG_TO_UPPER = {value[0]: key for key, value in ARCH_MAP.items()}
KNOWN_TAGS = sorted(TAG_TO_UPPER.keys(), key=len, reverse=True)
POLICIES = ["Sequential", "Streams", "Isolated", "MPS"]


def map_distribution(label: str) -> str:
    label = (label or "").strip().lower()
    if label in ("trace", "apollo", "appollo"):
        return "apollo"
    return label


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_metrics_in_obj(obj):
    found = []

    def is_metrics_dict(data):
        if not isinstance(data, dict):
            return False
        keys = {key.lower() for key in data.keys()}
        required = {"p50_latency", "p95_latency", "p99_latency", "throughput"}
        return required.issubset(keys)

    def visit(node):
        if isinstance(node, dict):
            if is_metrics_dict(node):
                found.append(node)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(obj)
    return found


def parse_flat_hp_lp(obj):
    if not isinstance(obj, dict):
        return None, None

    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    lp_metrics = {
        "p50_latency": safe_float(obj.get("p50-latency-0")),
        "p95_latency": safe_float(obj.get("p95-latency-0")),
        "p99_latency": safe_float(obj.get("p99-latency-0")),
        "throughput": safe_float(obj.get("throughput-0")),
    }
    hp_metrics = {
        "p50_latency": safe_float(obj.get("p50-latency-1")),
        "p95_latency": safe_float(obj.get("p95-latency-1")),
        "p99_latency": safe_float(obj.get("p99-latency-1")),
        "throughput": safe_float(obj.get("throughput-1")),
    }
    if None in lp_metrics.values() or None in hp_metrics.values():
        return None, None
    return hp_metrics, lp_metrics


def pick_hp_lp_metrics(obj, hp_tag: str, lp_tag: str):
    hp_metrics, lp_metrics = parse_flat_hp_lp(obj)
    if hp_metrics and lp_metrics:
        return hp_metrics, lp_metrics

    if isinstance(obj, dict):
        for hp_key, lp_key in [("hp", "lp"), ("HP", "LP"), ("model1", "model0"), (hp_tag, lp_tag)]:
            if hp_key in obj and lp_key in obj:
                hp_candidates = find_metrics_in_obj(obj[hp_key])
                lp_candidates = find_metrics_in_obj(obj[lp_key])
                if hp_candidates and lp_candidates:
                    return hp_candidates[0], lp_candidates[0]

    metrics = find_metrics_in_obj(obj)
    if len(metrics) >= 2:
        return metrics[0], metrics[1]
    if len(metrics) == 1:
        return metrics[0], None
    return None, None


def try_parse_text_metrics(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    pattern = re.compile(
        r"(hp|lp)?[^:\n]*p50[_\- ]latency[:=]\s*([0-9.]+).*?"
        r"p95[_\- ]latency[:=]\s*([0-9.]+).*?"
        r"p99[_\- ]latency[:=]\s*([0-9.]+).*?"
        r"throughput[:=]\s*([0-9.]+)",
        re.IGNORECASE | re.DOTALL,
    )
    hp_metrics = None
    lp_metrics = None
    for match in pattern.finditer(text):
        role = (match.group(1) or "").lower()
        metrics = {
            "p50_latency": float(match.group(2)),
            "p95_latency": float(match.group(3)),
            "p99_latency": float(match.group(4)),
            "throughput": float(match.group(5)),
        }
        if role == "hp" and hp_metrics is None:
            hp_metrics = metrics
        elif role == "lp" and lp_metrics is None:
            lp_metrics = metrics
        elif not role:
            if hp_metrics is None:
                hp_metrics = metrics
            elif lp_metrics is None:
                lp_metrics = metrics
        if hp_metrics and lp_metrics:
            break
    return hp_metrics, lp_metrics


def split_hp_lp_dirname(name: str):
    for hp_tag in KNOWN_TAGS:
        prefix = hp_tag + "_"
        if name.startswith(prefix):
            lp_tag = name[len(prefix):]
            if lp_tag in KNOWN_TAGS:
                return hp_tag, lp_tag

    for lp_tag in KNOWN_TAGS:
        suffix = "_" + lp_tag
        if name.endswith(suffix):
            hp_tag = name[: -len(suffix)]
            if hp_tag in KNOWN_TAGS:
                return hp_tag, lp_tag
    return None, None


def load_rates_from_config(config_path: Path, hp_tag: str, lp_tag: str):
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        return None
    hp_cfg = config.get(hp_tag, {}) or {}
    lp_cfg = config.get(lp_tag, {}) or {}
    hp_rps = hp_cfg.get("request_rate")
    lp_rps = lp_cfg.get("request_rate")
    hp_num_requests = hp_cfg.get("num_iterations")
    lp_num_requests = lp_cfg.get("num_iterations")
    hp_batch_size = hp_cfg.get("batch_size")
    lp_batch_size = lp_cfg.get("batch_size")
    return {
        "hp_rps": hp_rps,
        "lp_rps": lp_rps,
        "rps": hp_rps,
        "hp_num_requests": hp_num_requests,
        "lp_num_requests": lp_num_requests,
        "hp_batch_size": hp_batch_size,
        "lp_batch_size": lp_batch_size,
    }


def discover_run_dirs(batch_dir: Path):
    rps_dirs = sorted(
        [path for path in batch_dir.glob("rps_*") if path.is_dir()],
        key=lambda path: (
            0,
            int(path.name.replace("rps_", "")),
        )
        if path.name.replace("rps_", "").isdigit()
        else (1, path.name),
    )
    return rps_dirs or [batch_dir]


def gather_rows(results_dir: Path, policy: str):
    rows = []
    for hp_lp_dir in sorted(results_dir.glob("*_*")):
        if not hp_lp_dir.is_dir():
            continue

        hp_tag, lp_tag = split_hp_lp_dirname(hp_lp_dir.name)
        if not hp_tag or not lp_tag:
            candidates = list(hp_lp_dir.rglob("eval-*eval-*.log.json")) + list(hp_lp_dir.rglob("eval-*eval-*.log"))
            if candidates:
                match = re.search(r"eval-(.+)eval-(.+)\.log(?:\.json)?$", candidates[0].name)
                if match:
                    lp_tag, hp_tag = match.group(1), match.group(2)
        if not hp_tag or not lp_tag:
            continue

        hp_upper = TAG_TO_UPPER.get(hp_tag, hp_tag.upper())
        lp_upper = TAG_TO_UPPER.get(lp_tag, lp_tag.upper())
        policy_dir = hp_lp_dir / policy
        if not policy_dir.exists():
            continue

        for dist_dir in sorted([path for path in policy_dir.iterdir() if path.is_dir()]):
            distribution = map_distribution(dist_dir.name)
            for batch_dir in sorted(dist_dir.glob("batch_*")):
                if not batch_dir.is_dir():
                    continue

                for run_dir in discover_run_dirs(batch_dir):
                    config_path = run_dir / "config.yaml"
                    rates = load_rates_from_config(config_path, hp_tag, lp_tag) or {}
                    hp_batch_size = rates.get("hp_batch_size", "")
                    lp_batch_size = rates.get("lp_batch_size", "")
                    if hp_batch_size == "" or lp_batch_size == "":
                        batch_str = batch_dir.name.replace("batch_", "")
                        if "_" in batch_str:
                            hp_batch_size, lp_batch_size = batch_str.split("_", 1)
                        else:
                            hp_batch_size = batch_str
                            lp_batch_size = batch_str

                    eval_json = run_dir / f"eval-{lp_tag}eval-{hp_tag}.log.json"
                    eval_log_candidates = [
                        run_dir / f"eval-{lp_tag}eval-{hp_tag}.log",
                        run_dir / f"eval-{hp_tag}eval-{lp_tag}.log",
                    ]

                    hp_metrics = lp_metrics = None
                    if eval_json.exists():
                        obj = load_json(eval_json)
                        if obj is not None:
                            hp_metrics, lp_metrics = pick_hp_lp_metrics(obj, hp_tag, lp_tag)

                    if hp_metrics is None or lp_metrics is None:
                        for eval_log in eval_log_candidates:
                            hp_text, lp_text = try_parse_text_metrics(eval_log)
                            hp_metrics = hp_metrics or hp_text
                            lp_metrics = lp_metrics or lp_text
                            if hp_metrics and lp_metrics:
                                break

                    hp_cell = json.dumps(hp_metrics, ensure_ascii=False) if hp_metrics else ""
                    lp_cell = json.dumps(lp_metrics, ensure_ascii=False) if lp_metrics else ""
                    rows.append(
                        {
                            "policy": policy,
                            "models": f"{hp_upper}|{lp_upper}",
                            "bs": f"{hp_batch_size}|{lp_batch_size}",
                            "distribution": distribution,
                            "rps": rates.get("rps", ""),
                            "hp_rps": rates.get("hp_rps", ""),
                            "lp_rps": rates.get("lp_rps", ""),
                            "hp_num_requests": rates.get("hp_num_requests", ""),
                            "lp_num_requests": rates.get("lp_num_requests", ""),
                            "hp_p99_ms": hp_metrics.get("p99_latency") if hp_metrics else "",
                            "hp_throughput_rps": hp_metrics.get("throughput") if hp_metrics else "",
                            "lp_p99_ms": lp_metrics.get("p99_latency") if lp_metrics else "",
                            "lp_throughput_rps": lp_metrics.get("throughput") if lp_metrics else "",
                            "hp": hp_cell,
                            "lp": lp_cell,
                            "run_dir": str(run_dir.resolve()),
                        }
                    )

    def sort_key(row):
        def to_int(value):
            try:
                return int(value)
            except Exception:
                return 10**9

        return (
            to_int(row.get("rps")),
            row.get("models", ""),
            row.get("distribution", ""),
            row.get("policy", ""),
            row.get("run_dir", ""),
        )

    rows.sort(key=sort_key)
    return rows


def csv_fieldnames(include_policy: bool):
    base = [
        "models",
        "bs",
        "distribution",
        "rps",
        "hp_rps",
        "lp_rps",
        "hp_num_requests",
        "lp_num_requests",
        "hp_p99_ms",
        "hp_throughput_rps",
        "lp_p99_ms",
        "lp_throughput_rps",
        "hp",
        "lp",
        "run_dir",
    ]
    return ["policy"] + base if include_policy else base


def write_csv(rows, out_path: Path, include_policy: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames(include_policy)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(record)


def excel_col_name(index: int):
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def sanitize_sheet_name(name: str, used_names):
    sanitized = re.sub(r"[\[\]\:\*\?\/\\]", "_", name)[:31] or "Sheet"
    candidate = sanitized
    suffix = 1
    while candidate in used_names:
        raw = f"{sanitized[:28]}_{suffix}"
        candidate = raw[:31]
        suffix += 1
    used_names.add(candidate)
    return candidate


def build_cell_xml(row_idx: int, col_idx: int, value):
    cell_ref = f"{excel_col_name(col_idx)}{row_idx}"
    if value in ("", None):
        return f'<c r="{cell_ref}" t="inlineStr"><is><t></t></is></c>'
    if isinstance(value, bool):
        value = "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def build_sheet_xml(headers, rows):
    xml_rows = []
    header_cells = "".join(build_cell_xml(1, idx, header) for idx, header in enumerate(headers, start=1))
    xml_rows.append(f'<row r="1">{header_cells}</row>')
    for row_idx, row in enumerate(rows, start=2):
        cells = "".join(build_cell_xml(row_idx, idx, row.get(header, "")) for idx, header in enumerate(headers, start=1))
        xml_rows.append(f'<row r="{row_idx}">{cells}</row>')
    sheet_data = "".join(xml_rows)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def build_workbook_xml(sheet_names):
    sheets_xml = "".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets_xml}</sheets>"
        "</workbook>"
    )


def build_workbook_rels_xml(sheet_count: int):
    rels_xml = "".join(
        f'<Relationship Id="rId{idx}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels_xml}"
        "</Relationships>"
    )


def build_root_rels_xml():
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def build_content_types_xml(sheet_count: int):
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    overrides.extend(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    overrides_xml = "".join(overrides)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{overrides_xml}"
        "</Types>"
    )


def write_xlsx(rows, out_path: Path, include_policy: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = csv_fieldnames(include_policy)
    grouped = {}
    for row in rows:
        key = row.get("rps", "")
        sheet_key = f"rps_{key}" if key not in ("", None) else "rps_unknown"
        grouped.setdefault(sheet_key, []).append({header: row.get(header, "") for header in headers})
    if not grouped:
        grouped = {"rps_empty": []}

    ordered_sheet_keys = sorted(
        grouped.keys(),
        key=lambda key: int(key.replace("rps_", "")) if key.replace("rps_", "").isdigit() else key,
    )
    used_names = set()
    sheet_names = [sanitize_sheet_name(key, used_names) for key in ordered_sheet_keys]

    with ZipFile(out_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", build_content_types_xml(len(sheet_names)))
        archive.writestr("_rels/.rels", build_root_rels_xml())
        archive.writestr("xl/workbook.xml", build_workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", build_workbook_rels_xml(len(sheet_names)))
        for idx, sheet_key in enumerate(ordered_sheet_keys, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", build_sheet_xml(headers, grouped[sheet_key]))


def main():
    parser = argparse.ArgumentParser(
        description="Collect baseline results into CSV and XLSX (grouped by rps sheets)."
    )
    parser.add_argument("--results-dir", default="result", help="Root directory of baseline results")
    parser.add_argument(
        "--policy",
        default="Sequential",
        choices=["Sequential", "Streams", "Isolated", "MPS", "all"],
        help="Policy to export. Use 'all' to export per-policy and merged files.",
    )
    parser.add_argument("--out", default="baselines.csv", help="Output CSV path when --policy is a single value")
    parser.add_argument(
        "--xlsx",
        default="",
        help="Output XLSX path when --policy is a single value (default: same stem as --out)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    if args.policy == "all":
        merged_rows = []
        for policy in POLICIES:
            rows = gather_rows(results_dir, policy)
            csv_path = Path(f"baselines-{policy}.csv")
            xlsx_path = Path(f"baselines-{policy}.xlsx")
            write_csv(rows, csv_path, include_policy=False)
            write_xlsx(rows, xlsx_path, include_policy=False)
            merged_rows.extend(rows)

        merged_csv = Path("baselines_all.csv")
        merged_xlsx = Path("baselines_all.xlsx")
        write_csv(merged_rows, merged_csv, include_policy=True)
        write_xlsx(merged_rows, merged_xlsx, include_policy=True)
        print(
            "Wrote baselines-Sequential/Streams/Isolated/MPS.{csv,xlsx} and "
            "baselines_all.{csv,xlsx}"
        )
        return

    rows = gather_rows(results_dir, args.policy)
    csv_path = Path(args.out)
    xlsx_path = Path(args.xlsx) if args.xlsx else csv_path.with_suffix(".xlsx")
    write_csv(rows, csv_path, include_policy=False)
    write_xlsx(rows, xlsx_path, include_policy=False)
    print(f"Wrote {csv_path} and {xlsx_path} for policy={args.policy}")


if __name__ == "__main__":
    main()
