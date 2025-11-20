#!/usr/bin/env python3
"""
Parse MIMIR experiment logs in a directory and generate LaTeX tables for:
- ROC AUC
- TPR at FPR=0.001
- TPR at FPR=0.01

Usage:
  python python_scripts/generate_tables_from_logs.py logs/token_ref_1ref [-o results/token_ref_1ref_tables.tex]

Notes:
- Expects log filenames like: pythia-<size>__<dataset>.log
- Methods parsed (mapped to table rows):
    loss_threshold     -> Loss
    zlib_threshold     -> Zlib
    min_k_threshold    -> Min-K%
    min_k++_threshold  -> Min-K%++
    dc_pdd_threshold   -> DC-PDD
    ref-*threshold     -> Ref
    info_rmia_token_threshold -> Info-RMIA1 (if present). Info-RMIA2 left blank unless a matching key is found.
- Datasets included in tables (by slug in filenames):
    wikipedia_(en), github, pile_cc, pubmed_central, arxiv, dm_mathematics, hackernews
- Any other dataset (e.g., full_pile) is ignored.
- "Average" is computed as the simple mean across the seven datasets above, per method and model size, using available cells.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable


# Order and display names for datasets in the table
DATASET_GROUPS_ORDER: List[str] = [
    "wikipedia_(en)",
    "github",
    "pile_cc",
    "pubmed_central",
    "arxiv",
    "dm_mathematics",
    "hackernews",
]

DATASET_DISPLAY: Dict[str, str] = {
    "wikipedia_(en)": "Wikipedia",
    "github": "Github",
    "pile_cc": "Pile CC",
    "pubmed_central": "PubMed Central",
    "arxiv": "ArXiv",
    "dm_mathematics": "DM Mathematics",
    "hackernews": "HackerNews",
}


# Model sizes order (as appears in filenames and table columns)
MODEL_SIZES_ORDER: List[str] = ["160m", "1.4b", "2.8b", "6.9b"]
MODEL_DISPLAY: Dict[str, str] = {
    "160m": "160M",
    "1.4b": "1.4B",
    "2.8b": "2.8B",
    "6.9b": "6.9B",
}


# Map method keys that appear in logs to table row names
METHOD_KEY_TO_ROW: Dict[str, str] = {
    "loss_threshold": "Loss",
    "zlib_threshold": "Zlib",
    "min_k_threshold": "Min-K%",
    "min_k++_threshold": "Min-K%++",
    "dc_pdd_threshold": "DC-PDD",
    # "recall_threshold": "ReCaLL",  # intentionally excluded from output tables
    "info_rmia_token_threshold": "Info-RMIA1",  # If present; Info-RMIA2 left blank unless detected explicitly
}

# Special handler: any method starting with 'ref-' and ending with '_threshold' -> 'Ref'
REF_METHOD_REGEX = re.compile(r"^ref-.*_threshold$")


# Regex to parse metric lines with AUC and TPR dict. Example line:
# loss_threshold ROC AUC: 0.514019, PR AUC: 0.5138, tpr_at_low_fpr: {0.001: 0.0, 0.01: 0.007}
METRIC_LINE_RE = re.compile(
    r"^(?P<method>[A-Za-z0-9_:+\-\.]+)\s+ROC AUC:\s*(?P<auc>\d*\.\d+|\d+)(?:,.*?tpr_at_low_fpr:\s*\{\s*0\.001:\s*(?P<tpr001>\d*\.\d+|\d+),\s*0\.01:\s*(?P<tpr01>\d*\.\d+|\d+)\s*\})?",
    re.IGNORECASE,
)

# Fallback for summary-only lines like: "loss_threshold roc_auc: 0.514"
SUMMARY_AUC_RE = re.compile(
    r"^(?P<method>[A-Za-z0-9_:+\-\.]+)\s+roc_auc:\s*(?P<auc>\d*\.\d+|\d+)",
    re.IGNORECASE,
)


def parse_filename(fname: str) -> Optional[Tuple[str, str]]:
    """Return (model_size_slug, dataset_slug) from filename like 'pythia-1.4b__arxiv.log'."""
    base = os.path.basename(fname)
    if not base.startswith("pythia-") or not base.endswith(".log"):
        return None
    try:
        middle = base[len("pythia-") : -len(".log")]
        size, dataset = middle.split("__", 1)
        return size, dataset
    except Exception:
        return None


def method_to_row(method_key: str) -> Optional[str]:
    key = method_key.strip()
    # Normalize for matching
    k = key.lower()
    
    # Direct mapping
    if key in METHOD_KEY_TO_ROW:
        return METHOD_KEY_TO_ROW[key]

    # Handle AllAttacks.REFERENCE_BASED patterns (including with model suffixes)
    if 'reference_based' in k or 'allattacks.reference_based' in k:
        return 'Ref'
    
    # Drop optional namespace prefixes like 'AllAttacks.'
    if '.' in key:
        parts = key.split('.')
        # Prefer the last segment as the method name
        key_core = parts[-1]
    else:
        key_core = key

    # Normalize core key for matching
    k_core = key_core.lower()

    # Heuristics for known methods
    if 'reference' in k_core or 'reference_based' in k_core:
        return 'Ref'
    if 'loss_threshold' in k_core or k_core == 'loss':
        return 'Loss'
    if 'zlib' in k_core:
        return 'Zlib'
    if 'min_k++' in k_core or 'min-k++' in k_core or 'min_k%++' in k_core:
        return 'Min-K%++'
    if 'min_k' in k_core or 'min-k' in k_core or 'min_k%' in k_core:
        return 'Min-K%'
    if 'dc_pdd' in k_core or 'dc-pdd' in k_core:
        return 'DC-PDD'
    if 'info_rmia_token' in k_core or 'info-rmia-token' in k_core:
        return 'Info-RMIA1'
    if REF_METHOD_REGEX.match(key):
        return 'Ref'
    # Unknown or unsupported method
    return None


def parse_log_file(
    path: str,
    method_mapper: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """Parse a single log file and return mapping: row_name -> (auc, tpr001, tpr01).

    Returns the LAST occurrence of each metric within the file for robustness.
    """
    results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return results

    for line in lines:
        sline = line.strip()
        m = METRIC_LINE_RE.search(sline)
        if not m:
            # Try fallback summary AUC format
            m2 = SUMMARY_AUC_RE.search(sline)
            if not m2:
                continue
            method_key = m2.group("method").strip()
            row = method_mapper(method_key) if method_mapper else method_to_row(method_key)
            if row is None:
                continue
            auc = float(m2.group("auc")) if m2.group("auc") is not None else None
            # Don't overwrite if we already saw full metrics with TPRs
            prev = results.get(row)
            if prev is None or prev[0] is None:
                results[row] = (auc, None, None)
            continue
        method_key = m.group("method").strip()
        row = method_mapper(method_key) if method_mapper else method_to_row(method_key)
        if row is None:
            continue
        auc = float(m.group("auc")) if m.group("auc") is not None else None
        tpr001 = float(m.group("tpr001")) if m.group("tpr001") is not None else None
        tpr01 = float(m.group("tpr01")) if m.group("tpr01") is not None else None
        results[row] = (auc, tpr001, tpr01)

    return results


def fmt_pct(x: Optional[float]) -> str:
    """Format a fraction as percentage with one decimal (e.g., 0.514 -> '51.4'). Empty if None."""
    if x is None:
        return ""
    return f"{x * 100:.1f}"


def compute_average(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_bests(
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    metric_key: str,
    rows_order: List[str],
    model_sizes: List[str],
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Dict[str, Optional[float]]]]:
    """Compute best value per dataset and model size across methods; also for Average.

    Returns a tuple of two dictionaries:
    1. bests_all: bests[dataset_or_Average][model_size] = max_value (float or None) across ALL methods.
    2. bests_subset: bests[dataset_or_Average][model_size] = max_value (float or None) across SUBSET of methods.
    """
    subset_rows = ["Ref", "Info-RMIA1", "Info-RMIA2"]

    bests_all: Dict[str, Dict[str, Optional[float]]] = defaultdict(lambda: {s: None for s in model_sizes})
    bests_subset: Dict[str, Dict[str, Optional[float]]] = defaultdict(lambda: {s: None for s in model_sizes})

    # Per-dataset bests
    for ds in DATASET_GROUPS_ORDER:
        for size in model_sizes:
            max_val_all: Optional[float] = None
            max_val_subset: Optional[float] = None
            for r in rows_order:
                v = data.get(r, {}).get(ds, {}).get(size, {}).get(metric_key)
                if v is None:
                    continue
                if max_val_all is None or v > max_val_all:
                    max_val_all = v
                if r in subset_rows:
                    if max_val_subset is None or v > max_val_subset:
                        max_val_subset = v
            bests_all[ds][size] = max_val_all
            bests_subset[ds][size] = max_val_subset

    # Average bests (across datasets per method and model size)
    for size in model_sizes:
        max_avg_all: Optional[float] = None
        max_avg_subset: Optional[float] = None
        for r in rows_order:
            vals: List[Optional[float]] = []
            for ds in DATASET_GROUPS_ORDER:
                vals.append(data.get(r, {}).get(ds, {}).get(size, {}).get(metric_key))
            avg = compute_average(vals)
            if avg is None:
                continue
            if max_avg_all is None or avg > max_avg_all:
                max_avg_all = avg
            if r in subset_rows:
                if max_avg_subset is None or avg > max_avg_subset:
                    max_avg_subset = avg
        bests_all["Average"][size] = max_avg_all
        bests_subset["Average"][size] = max_avg_subset

    return bests_all, bests_subset


def build_table_body(
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    metric_key: str,
    bests_all: Dict[str, Dict[str, Optional[float]]],
    bests_subset: Dict[str, Dict[str, Optional[float]]],
    top_groups: List[str],
    bottom_groups: List[str],
    rows_order: List[str],
    model_sizes: List[str],
    underline_subset: bool,
    bold_all: bool,
) -> str:
    """Build LaTeX rows for one metric with single-line rows per method."""

    def cell_text(row_name: str, ds_or_avg: str, size: str, value: Optional[float]) -> str:
        s = fmt_pct(value)
        if value is None:
            return s

        is_best_all = False
        best_all = bests_all.get(ds_or_avg, {}).get(size)
        if best_all is not None and abs(value - best_all) < 1e-9:
            is_best_all = True

        is_best_subset = False
        if underline_subset and row_name in ["Ref", "Info-RMIA1", "Info-RMIA2"]:
            best_subset = bests_subset.get(ds_or_avg, {}).get(size)
            if best_subset is not None and abs(value - best_subset) < 1e-9:
                is_best_subset = True

        # Apply formatting based on mode
        if bold_all and underline_subset:
            # Default mode: both bold and underline
            if is_best_all and is_best_subset:
                return rf"\underline{{\textbf{{{s}}}}}"
            if is_best_all:
                return rf"\textbf{{{s}}}"
            if is_best_subset:
                return rf"\underline{{{s}}}"
        elif underline_subset and not bold_all:
            # _1sameref mode: only underline, no bold
            if is_best_all:  # In _1sameref, "best all" means best among the 3 methods
                return rf"\underline{{{s}}}"
        elif bold_all and not underline_subset:
            # Bold only mode
            if is_best_all:
                return rf"\textbf{{{s}}}"
        return s

    def row_line(row_name: str, groups: List[str], include_average: bool) -> str:
        display_row = row_name
        if row_name == "Min-K%":
            display_row = r"Min-K\%"
        elif row_name == "Min-K%++":
            display_row = r"Min-K\%++"

        pieces: List[str] = []
        for ds in groups:
            for size in model_sizes:
                v = data.get(row_name, {}).get(ds, {}).get(size, {}).get(metric_key)
                pieces.append(cell_text(row_name, ds, size, v))
        if include_average:
            for size in model_sizes:
                vals = [data.get(row_name, {}).get(ds, {}).get(size, {}).get(metric_key) for ds in DATASET_GROUPS_ORDER]
                avg = compute_average(vals)
                pieces.append(cell_text(row_name, "Average", size, avg))
        return display_row.ljust(12) + " & " + " & ".join(pieces) + r" \\"  # one \ per row

    out_lines: List[str] = []

    # Top half rows (one line per method)
    for r in rows_order:
        if r in ("Info-RMIA1", "Info-RMIA2"):
            out_lines.append(r"\rowcolor{RowGray}")
        out_lines.append(row_line(r, top_groups, include_average=False))

    out_lines.append(r"\toprule")

    # Bottom half rows (one line per method, includes Average block)
    for r in rows_order:
        if r in ("Info-RMIA1", "Info-RMIA2"):
            out_lines.append(r"\rowcolor{RowGray}")
        out_lines.append(row_line(r, bottom_groups, include_average=True))

    return "\n".join(out_lines)


def build_table_tex(
    title: str,
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    metric_key: str,
    rows_order: List[str],
    model_sizes: List[str],
    underline_subset: bool,
    bold_all: bool,
    use_resizebox: bool,
) -> str:
    top_groups = ["wikipedia_(en)", "github", "pile_cc", "pubmed_central"]
    bottom_groups = ["arxiv", "dm_mathematics", "hackernews"]

    num_cols_per_group = len(model_sizes)
    num_top_groups = len(top_groups)
    num_bottom_groups = len(bottom_groups) + 1  # +1 for Average

    def make_header(groups: List[str], include_average: bool) -> str:
        model_size_labels = " & ".join([MODEL_DISPLAY.get(s, s) for s in model_sizes])
        
        main_headers = "& " + " & ".join([rf"\multicolumn{{{num_cols_per_group}}}{{c}}{{\textbf{{{DATASET_DISPLAY.get(ds, ds)}}}}}" for ds in groups])
        if include_average:
            main_headers += rf" & \multicolumn{{{num_cols_per_group}}}{{c}}{{\textbf{{Average}}}}"
        main_headers += r" \\"

        cmidrules = " ".join([rf"\cmidrule(lr){{{2 + i*num_cols_per_group}-{2 + i*num_cols_per_group + num_cols_per_group - 1}}}" for i in range(len(groups) + (1 if include_average else 0))])

        repeated_model_sizes = " & ".join([model_size_labels] * (len(groups) + (1 if include_average else 0)))
        
        return (
            f"{main_headers}\n"
            f"{cmidrules}\n"
            r"\textbf{Method}"
            f"& {repeated_model_sizes} \\\\\n"
            r"\midrule"
        )

    header_top = make_header(top_groups, include_average=False)
    header_bottom = make_header(bottom_groups, include_average=True)

    bests_all, bests_subset = compute_bests(data, metric_key, rows_order, model_sizes)
    body = build_table_body(data, metric_key, bests_all, bests_subset, top_groups, bottom_groups, rows_order, model_sizes, underline_subset, bold_all)

    # For the bottom half labels, the body already contains a second \toprule followed by bottom groups and averages.
    # To match the provided template structure (two halves), we wrap the body with appropriate headers and a split.
    # We'll inject the bottom header by replacing the first occurrence of the internal \toprule delimiter inside body.

    # Split body at the first '\\toprule' (inserted between halves)
    parts = body.split("\\toprule")
    body_top = body
    body_bottom = ""
    if len(parts) == 2:
        body_top, body_bottom = parts

    num_total_cols = 1 + num_cols_per_group * max(num_top_groups, num_bottom_groups)

    if body_bottom.strip():
        table_content = fr"""
\begin{{tabular}}{{l *{{{num_total_cols-1}}}{{c}}}}
\toprule
{header_top}
{body_top}
\toprule
{header_bottom}
{body_bottom}
\bottomrule
\end{{tabular}}
"""
    else:
        table_content = fr"""
\begin{{tabular}}{{l *{{{num_total_cols-1}}}{{c}}}}
\toprule
{header_top}
{body}
\bottomrule
\end{{tabular}}
"""

    if use_resizebox:
        return fr"""
% -------- {title} --------
\resizebox{{\textwidth}}{{!}}{{%
{table_content.strip()}%
}}
""".strip()
    else:
        return fr"""
% -------- {title} --------
{table_content.strip()}
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from MIMIR logs.")
    parser.add_argument("log_dir", help="Path to directory containing .log files (e.g., logs/token_ref_1ref)")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .tex path. Defaults to results/<dirname>_tables.tex",
        default=None,
    )
    parser.add_argument(
        "--irmia2-dir",
        dest="irmia2_dir",
        default=None,
        help="Optional directory holding logs for Info-RMIA2 (min-k%). If omitted, attempts '<log_dir>_mink' if it exists.",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        raise SystemExit(f"Not a directory: {log_dir}")

    # --- Special handling for '_1sameref' directories ---
    is_sameref_mode = "_1sameref" in log_dir
    if is_sameref_mode:
        rows_order = [
            "Loss", "Zlib", "Min-K%", "Min-K%++", "DC-PDD", 
            "Ref", "Info-RMIA1", "Info-RMIA2"
        ]
        model_sizes = MODEL_SIZES_ORDER  # Include all model sizes including 160m
        underline_subset = True
        bold_all = True  # Enable bolding like _1ref
        use_resizebox = True  # Bring back resizebox
        table_env = "table*"  # Use table* with resizebox
        table_formatting = "\\begin{center}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\renewcommand{\\arraystretch}{1.15}\n\n"
    else:
        rows_order = [
            "Loss", "Zlib", "Min-K%", "Min-K%++", "DC-PDD", 
            "Ref", "Info-RMIA1", "Info-RMIA2"
        ]
        model_sizes = MODEL_SIZES_ORDER
        underline_subset = True
        bold_all = True
        use_resizebox = True
        table_env = "table*"
        table_formatting = "\\begin{center}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\renewcommand{\\arraystretch}{1.15}\n\n"
    # ----------------------------------------------------

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.basename(os.path.normpath(log_dir))
        out_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_tables.tex")

    # Data structure: row -> dataset -> model_size -> {auc, tpr001, tpr01}
    data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"auc": None, "tpr001": None, "tpr01": None}))
    )

    # For _1sameref mode, parse methods from different directories
    if is_sameref_mode:
        # First, parse OTHER methods (not Ref, Info-RMIA1, Info-RMIA2) from _1ref directory
        ref_1ref_dir = log_dir.replace("_1sameref", "_1ref")
        if os.path.isdir(ref_1ref_dir):
            for fname in os.listdir(ref_1ref_dir):
                if not fname.endswith(".log"):
                    continue
                parsed = parse_filename(fname)
                if not parsed:
                    continue
                model_size, dataset = parsed
                if dataset not in DATASET_GROUPS_ORDER:
                    continue
                if model_size not in model_sizes:
                    continue
                path = os.path.join(ref_1ref_dir, fname)
                metrics = parse_log_file(path)
                for row_name, (auc, tpr001, tpr01) in metrics.items():
                    if row_name not in rows_order:
                        continue
                    # Skip Ref, Info-RMIA1, Info-RMIA2 - these come from _1sameref directories
                    if row_name in ["Ref", "Info-RMIA1", "Info-RMIA2"]:
                        continue
                    data[row_name][dataset][model_size]["auc"] = auc
                    data[row_name][dataset][model_size]["tpr001"] = tpr001
                    data[row_name][dataset][model_size]["tpr01"] = tpr01

        # Then, parse Ref and Info-RMIA1 from _1sameref directory
        for fname in os.listdir(log_dir):
            if not fname.endswith(".log"):
                continue
            parsed = parse_filename(fname)
            if not parsed:
                continue
            model_size, dataset = parsed
            if dataset not in DATASET_GROUPS_ORDER:
                continue
            if model_size not in model_sizes:
                continue

            path = os.path.join(log_dir, fname)
            metrics = parse_log_file(path)
            for row_name, (auc, tpr001, tpr01) in metrics.items():
                if row_name not in rows_order:
                    continue
                # Only take Ref and Info-RMIA1 from this directory
                if row_name in ["Ref", "Info-RMIA1"]:
                    data[row_name][dataset][model_size]["auc"] = auc
                    data[row_name][dataset][model_size]["tpr001"] = tpr001
                    data[row_name][dataset][model_size]["tpr01"] = tpr01

        # Finally, parse Info-RMIA2 from _1sameref_mink directory
        ref_1sameref_mink_dir = log_dir.rstrip("/") + "_mink"
        if os.path.isdir(ref_1sameref_mink_dir):
            def irmia2_mapper_sameref(method_key: str) -> Optional[str]:
                k = method_key.lower()
                if "info_rmia_token_threshold" in k or "info-rmia-token" in k or "info_rmia_token" in k:
                    return "Info-RMIA2"
                return None

            for fname in os.listdir(ref_1sameref_mink_dir):
                if not fname.endswith(".log"):
                    continue
                parsed = parse_filename(fname)
                if not parsed:
                    continue
                model_size, dataset = parsed
                if dataset not in DATASET_GROUPS_ORDER:
                    continue
                if model_size not in model_sizes:
                    continue
                path = os.path.join(ref_1sameref_mink_dir, fname)
                metrics = parse_log_file(path, method_mapper=irmia2_mapper_sameref)
                for row_name, (auc, tpr001, tpr01) in metrics.items():
                    if row_name not in rows_order:
                        continue
                    # Only take Info-RMIA2 from this directory
                    if row_name == "Info-RMIA2":
                        data[row_name][dataset][model_size]["auc"] = auc
                        data[row_name][dataset][model_size]["tpr001"] = tpr001
                        data[row_name][dataset][model_size]["tpr01"] = tpr01
    else:
        # Normal mode: parse all logs from the main directory
        for fname in os.listdir(log_dir):
            if not fname.endswith(".log"):
                continue
            parsed = parse_filename(fname)
            if not parsed:
                continue
            model_size, dataset = parsed
            if dataset not in DATASET_GROUPS_ORDER:
                # Ignore datasets not in the table
                continue
            if model_size not in model_sizes:
                # Ignore unknown or excluded model sizes
                continue

            path = os.path.join(log_dir, fname)
            metrics = parse_log_file(path)
            for row_name, (auc, tpr001, tpr01) in metrics.items():
                if row_name not in rows_order:
                    continue
                data[row_name][dataset][model_size]["auc"] = auc
                data[row_name][dataset][model_size]["tpr001"] = tpr001
                data[row_name][dataset][model_size]["tpr01"] = tpr01

        # Optionally parse Info-RMIA2 from a secondary directory
        irmia2_dir = args.irmia2_dir
        if irmia2_dir is None:
            guess = log_dir.rstrip("/") + "_mink"
            if os.path.isdir(guess):
                irmia2_dir = guess
        if irmia2_dir and os.path.isdir(irmia2_dir):
            def irmia2_mapper(method_key: str) -> Optional[str]:
                k = method_key.lower()
                # Only accept Info-RMIA token threshold from this directory
                if "info_rmia_token_threshold" in k or "info-rmia-token" in k or "info_rmia_token" in k:
                    return "Info-RMIA2"
                return None

            for fname in os.listdir(irmia2_dir):
                if not fname.endswith(".log"):
                    continue
                parsed = parse_filename(fname)
                if not parsed:
                    continue
                model_size, dataset = parsed
                if dataset not in DATASET_GROUPS_ORDER or model_size not in model_sizes:
                    continue
                path = os.path.join(irmia2_dir, fname)
                metrics = parse_log_file(path, method_mapper=irmia2_mapper)
                for row_name, (auc, tpr001, tpr01) in metrics.items():
                    data[row_name][dataset][model_size]["auc"] = auc
                    data[row_name][dataset][model_size]["tpr001"] = tpr001
                    data[row_name][dataset][model_size]["tpr01"] = tpr01

    # Build three tables
    roc_auc_table = build_table_tex("ROC AUC", data, "auc", rows_order, model_sizes, underline_subset, bold_all, use_resizebox)
    tpr001_table = build_table_tex("TPR@FPR=0.001", data, "tpr001", rows_order, model_sizes, underline_subset, bold_all, use_resizebox)
    tpr01_table = build_table_tex("TPR@FPR=0.01", data, "tpr01", rows_order, model_sizes, underline_subset, bold_all, use_resizebox)

    # Compose full LaTeX content (tables only; caption/label intentionally omitted as per instruction)
    # Decide caption text by metric
    def caption_for(metric: str) -> str:
        if metric == "auc":
            return (
                "AUC results on MIMIR benchmark with deduped Pythia models. The \\emph{Neighbor} method is not included due to its computational complexity and relatively inferior performance reported in prior works. \\emph{ReCaLL} is not included for reasons in Appendix~\\ref{app:recall}. \\emph{Ref} method is evaluated using the checkpoint of Pythia-160m after the first step as the reference model. Our method (\\emph{Info-RMIA}) is the token-based Info-RMIA that does not require additional population data. Info-RMIA1 uses averaging to aggregate, while Info-RMIA2 uses min-k\\%, using the same hyperparameter $k$ as \\emph{Min-K\\%} and \\emph{Min-K\\%++}. Both are shaded for emphasis."
            )
        if metric == "tpr001":
            return "TPR at FPR=0.001 (TPR@0.001 FPR) on MIMIR benchmark with deduped Pythia models."
        if metric == "tpr01":
            return "TPR at FPR=0.01 (TPR@0.01 FPR) on MIMIR benchmark with deduped Pythia models."
        return metric

    def label_for(metric: str) -> str:
        if metric == "auc":
            return "table:mimir"
        if metric == "tpr001":
            return "table:mimir-tpr001"
        if metric == "tpr01":
            return "table:mimir-tpr01"
        return f"table:mimir-{metric}"

    def build_full_table_string(metric_name: str, table_code: str) -> str:
        caption = caption_for(metric_name)
        label = label_for(metric_name)
        end_formatting = "\\end{center}\n" if table_formatting else ""
        return (
            f"\\begin{{{table_env}}}[ht!]\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{table_formatting}"
            f"{table_code}\n"
            f"{end_formatting}"
            f"\\end{{{table_env}}}\n"
        )

    tex_content = (
        "% Auto-generated LaTeX tables from logs in: "
        + os.path.abspath(log_dir)
        + "\n\n"
        + build_full_table_string("auc", roc_auc_table) + "\n"
        + build_full_table_string("tpr001", tpr001_table) + "\n"
        + build_full_table_string("tpr01", tpr01_table)
    )

    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"Saved tables to: {out_path}")


if __name__ == "__main__":
    main()
