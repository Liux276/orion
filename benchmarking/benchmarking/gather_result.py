#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

KNOWN_TAGS = ["mobilenet_v2", "resnet101", "resnet50", "bert"]
TAG_TO_UPPER = {
    "resnet50": "RESNET50",
    "resnet101": "RESNET101",
    "mobilenet_v2": "MOBILENET_V2",
    "bert": "BERT",
}


def map_distribution(label: str) -> str:
    label = (label or "").strip().lower()
    if label in {"trace", "apollo", "appollo"}:
        return "apollo"
    return label


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def metrics_to_cell(metrics):
    if not isinstance(metrics, dict) or not metrics:
        return ""
    cleaned = {}
    for key in ("p50_latency", "p95_latency", "p99_latency", "throughput"):
        number = safe_float(metrics.get(key))
        if number is not None:
            cleaned[key] = number
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True) if cleaned else ""


def summary_to_row(summary, summary_path: Path):
    hp_metrics = summary.get("hp") if isinstance(summary.get("hp"), dict) else {}
    lp_metrics = summary.get("lp") if isinstance(summary.get("lp"), dict) else {}
    return {
        "strategy": summary.get("strategy", ""),
        "models": summary.get("models", ""),
        "bs": summary.get("bs", ""),
        "distribution": map_distribution(summary.get("distribution", "")),
        "rps": summary.get("rps", ""),
        "hp_rps": summary.get("hp_rps", summary.get("rps", "")),
        "lp_rps": summary.get("lp_rps", ""),
        "hp_num_requests": summary.get("hp_num_requests", ""),
        "lp_num_requests": summary.get("lp_num_requests", ""),
        "hp_p99_ms": summary.get("hp_p99_ms", ""),
        "hp_throughput_rps": summary.get("hp_throughput_rps", ""),
        "lp_p99_ms": summary.get("lp_p99_ms", ""),
        "lp_throughput_rps": summary.get("lp_throughput_rps", ""),
        "hp": metrics_to_cell(hp_metrics),
        "lp": metrics_to_cell(lp_metrics),
        "run_dir": summary.get("run_dir", str(summary_path.parent.resolve())),
    }


def collect_from_summaries(root: Path, algo: str):
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue
        strategy = str(summary.get("strategy", "")).strip().lower()
        if algo != "all" and strategy != algo:
            continue
        rows.append(summary_to_row(summary, summary_path))
    return rows


def split_pair_stub_be_hp(filename_or_stub: str):
    base = Path(filename_or_stub).name
    for pref in ("run_orion_", "run_reef_", "cmd_orion_", "cmd_reef_", "elapsed_orion_", "elapsed_reef_"):
        if base.startswith(pref):
            base = base[len(pref):]
    while True:
        old = base
        for suf in (".stdout", ".stderr", ".log", ".json", ".txt"):
            if base.endswith(suf):
                base = base[: -len(suf)]
        if base == old:
            break
    for be_tag in sorted(KNOWN_TAGS, key=len, reverse=True):
        if base.startswith(be_tag + "_"):
            hp_tag = base[len(be_tag) + 1 :]
            if hp_tag in KNOWN_TAGS:
                return be_tag, hp_tag
    tag_pat = "|".join(map(re.escape, KNOWN_TAGS))
    match = re.match(rf"^({tag_pat})_({tag_pat})$", base)
    if match:
        return match.group(1), match.group(2)
    return None, None


def convert_latency_to_ms(raw_value: Optional[float], model_tag: str):
    if raw_value is None:
        return None
    return raw_value if model_tag == "bert" else raw_value * 1000.0


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    lat_re = re.compile(
        r"Client\s+([01])\s+finished!\s+p50:\s*([0-9.]+)\s*sec,\s*p95:\s*([0-9.]+)\s*sec,\s*p99:\s*([0-9.]+)\s*sec",
        re.IGNORECASE,
    )
    latencies = {}
    for match in lat_re.finditer(text):
        cid = int(match.group(1))
        latencies[cid] = {
            "p50_latency": float(match.group(2)),
            "p95_latency": float(match.group(3)),
            "p99_latency": float(match.group(4)),
        }

    iters = {}
    iter_re = re.compile(r"=======\s+Client\s+([01])\s+has\s+done\s+(\d+)\s+iterations", re.IGNORECASE)
    for match in iter_re.finditer(text):
        iters[int(match.group(1))] = int(match.group(2))

    max_batch_idx = {}
    batch_re = re.compile(r"Client\s+([01]).*?batch_idx\s+is\s+(\d+)", re.IGNORECASE)
    for match in batch_re.finditer(text):
        cid = int(match.group(1))
        idx = int(match.group(2))
        if cid not in max_batch_idx or idx > max_batch_idx[cid]:
            max_batch_idx[cid] = idx

    overall = None
    for match in re.finditer(r"Total loop took\s+([0-9.]+)\s*sec", text, flags=re.IGNORECASE):
        overall = float(match.group(1))
    if overall is None:
        for match in re.finditer(r"Total time is\s+([0-9.]+)", text, flags=re.IGNORECASE):
            overall = float(match.group(1))

    total_by_client = {}
    for match in re.finditer(
        r"Client\s+([01]),\s*Total loop took\s+([0-9.]+)\s*sec", text, flags=re.IGNORECASE
    ):
        total_by_client[int(match.group(1))] = float(match.group(2))

    return {
        "latencies": latencies,
        "iters": iters,
        "max_batch_idx": max_batch_idx,
        "total": overall,
        "total_by_client": total_by_client,
    }


def throughput_from_info(info, cid: int):
    if cid in info["max_batch_idx"]:
        count = info["max_batch_idx"][cid] + 1
    else:
        count = info["iters"].get(cid)
    if not count or count <= 10:
        return None

    total_time = info["total"] or info["total_by_client"].get(cid)
    if not total_time or total_time <= 0:
        return None
    return (float(count) - 10.0) / float(total_time)


def load_rates_from_config(config_path: Path):
    payload = load_json(config_path)
    if not isinstance(payload, list) or len(payload) < 2:
        return {}
    lp_cfg, hp_cfg = payload[0], payload[1]
    lp_args = lp_cfg.get("args", {}) if isinstance(lp_cfg, dict) else {}
    hp_args = hp_cfg.get("args", {}) if isinstance(hp_cfg, dict) else {}
    return {
        "hp_rps": hp_args.get("rps", ""),
        "lp_rps": lp_args.get("rps", ""),
        "rps": hp_args.get("rps", ""),
        "hp_num_requests": hp_cfg.get("num_iters", ""),
        "lp_num_requests": lp_cfg.get("num_iters", ""),
        "hp_batch_size": hp_args.get("batchsize", ""),
        "lp_batch_size": lp_args.get("batchsize", ""),
    }


def collect_legacy(root: Path, algo: str):
    rows = []
    algo_dir = root / algo
    if not algo_dir.exists():
        return rows

    for batch_dir in sorted(algo_dir.glob("batch_*")):
        for dist_dir in sorted([path for path in batch_dir.iterdir() if path.is_dir()]):
            distribution = map_distribution(dist_dir.name)
            for log_path in sorted(dist_dir.glob(f"run_{algo}_*.stdout.log")):
                be_tag, hp_tag = split_pair_stub_be_hp(log_path.name)
                if not be_tag or not hp_tag:
                    continue

                rates = load_rates_from_config(dist_dir / f"{be_tag}_{hp_tag}.json")
                hp_batch_size = rates.get("hp_batch_size", batch_dir.name.replace("batch_", ""))
                lp_batch_size = rates.get("lp_batch_size", batch_dir.name.replace("batch_", ""))
                info = parse_log(log_path)

                hp_raw = info["latencies"].get(1)
                lp_raw = info["latencies"].get(0)
                hp_metrics = None
                lp_metrics = None
                if hp_raw:
                    hp_metrics = {
                        "p50_latency": convert_latency_to_ms(hp_raw.get("p50_latency"), hp_tag),
                        "p95_latency": convert_latency_to_ms(hp_raw.get("p95_latency"), hp_tag),
                        "p99_latency": convert_latency_to_ms(hp_raw.get("p99_latency"), hp_tag),
                        "throughput": throughput_from_info(info, 1),
                    }
                if lp_raw:
                    lp_metrics = {
                        "p50_latency": convert_latency_to_ms(lp_raw.get("p50_latency"), be_tag),
                        "p95_latency": convert_latency_to_ms(lp_raw.get("p95_latency"), be_tag),
                        "p99_latency": convert_latency_to_ms(lp_raw.get("p99_latency"), be_tag),
                        "throughput": throughput_from_info(info, 0),
                    }

                rows.append(
                    {
                        "strategy": algo.upper(),
                        "models": f"{TAG_TO_UPPER.get(hp_tag, hp_tag.upper())}|{TAG_TO_UPPER.get(be_tag, be_tag.upper())}",
                        "bs": f"{hp_batch_size}|{lp_batch_size}",
                        "distribution": distribution,
                        "rps": rates.get("rps", ""),
                        "hp_rps": rates.get("hp_rps", ""),
                        "lp_rps": rates.get("lp_rps", ""),
                        "hp_num_requests": rates.get("hp_num_requests", ""),
                        "lp_num_requests": rates.get("lp_num_requests", ""),
                        "hp_p99_ms": safe_float((hp_metrics or {}).get("p99_latency")) or "",
                        "hp_throughput_rps": safe_float((hp_metrics or {}).get("throughput")) or "",
                        "lp_p99_ms": safe_float((lp_metrics or {}).get("p99_latency")) or "",
                        "lp_throughput_rps": safe_float((lp_metrics or {}).get("throughput")) or "",
                        "hp": metrics_to_cell(hp_metrics),
                        "lp": metrics_to_cell(lp_metrics),
                        "run_dir": str(dist_dir.resolve()),
                    }
                )
    return rows


def collect(root: Path, algo: str):
    rows = collect_from_summaries(root, algo)
    if rows:
        rows.sort(key=sort_key)
        return rows

    if algo == "all":
        legacy_rows = collect_legacy(root, "orion") + collect_legacy(root, "reef")
    else:
        legacy_rows = collect_legacy(root, algo)
    legacy_rows.sort(key=sort_key)
    return legacy_rows


def csv_fieldnames():
    return [
        "strategy",
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
        row.get("strategy", ""),
        row.get("run_dir", ""),
    )


def write_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = csv_fieldnames()
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
        candidate = f"{sanitized[:28]}_{suffix}"[:31]
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
        cells = "".join(
            build_cell_xml(row_idx, idx, row.get(header, ""))
            for idx, header in enumerate(headers, start=1)
        )
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


def write_xlsx(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = csv_fieldnames()
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
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Collect Orion/REEF benchmark summaries into CSV and XLSX grouped by rps sheets."
    )
    parser.add_argument(
        "--root",
        default=str(script_dir / "result"),
        help="结果根目录（run.py 默认输出目录）",
    )
    parser.add_argument(
        "--algo",
        default="all",
        choices=["all", "orion", "reef"],
        help="读取哪个策略的结果；all 表示合并 Orion 和 REEF",
    )
    parser.add_argument("--out", default="baselines.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--xlsx",
        default="",
        help="输出 XLSX 路径，默认与 --out 同名但后缀为 .xlsx",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    rows = collect(root, args.algo.lower())
    csv_path = Path(args.out)
    xlsx_path = Path(args.xlsx) if args.xlsx else csv_path.with_suffix(".xlsx")
    write_csv(rows, csv_path)
    write_xlsx(rows, xlsx_path)
    print(f"Wrote {csv_path} and {xlsx_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
