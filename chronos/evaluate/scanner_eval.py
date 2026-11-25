"""扫描器评测: 行级 precision / recall / F1, 并统计漏报与误报行。"""
from __future__ import annotations

import os
from typing import Dict, List, Set

from ..scan.wallclock import Finding, scan_file


def load_labels(path: str) -> dict:
    ns: dict = {}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    return {"bad": ns.get("bad_lines", {}), "good": set(ns.get("good_lines", set()))}


def evaluate_scan(corpus_dir: str, labels_path: str = None) -> dict:
    """若 labels_path 给单个文件, 则只评该文件; 否则自动为目录里每个
    `<name>.py` 找同名 `<name>.labels.py` (逐文件行级评估)。"""
    if labels_path is not None:
        return _eval_single(corpus_dir, labels_path)
    # 目录模式: 每个 .py 找自己的 .labels.py
    targets = []
    for fn in sorted(os.listdir(corpus_dir)):
        if fn.endswith(".py") and not fn.endswith(".labels.py"):
            lab = os.path.join(corpus_dir, fn[:-3] + ".labels.py")
            if os.path.exists(lab):
                targets.append((os.path.join(corpus_dir, fn), lab))
    all_findings = []
    bad_all: Dict[str, Dict[int, str]] = {}
    good_all: Dict[str, Set[int]] = {}
    for src, lab in targets:
        all_findings.extend(scan_file(src))
        L = load_labels(lab)
        bad_all[src] = L["bad"]
        good_all[src] = L["good"]
    return _score(all_findings, bad_all, good_all)


def _eval_single(src_or_dir: str, labels_path: str) -> dict:
    labels = load_labels(labels_path)
    if os.path.isdir(src_or_dir):
        findings = []
        for fn in sorted(os.listdir(src_or_dir)):
            if fn.endswith(".py") and not fn.endswith(".labels.py"):
                findings.extend(scan_file(os.path.join(src_or_dir, fn)))
        target = os.path.join(src_or_dir, "?")
    else:
        findings = scan_file(src_or_dir)
        target = src_or_dir
    return _score(findings, {target: labels["bad"]}, {target: labels["good"]})


def _score(findings: List[Finding], bad_by_file, good_by_file) -> dict:
    by_file: Dict[str, List[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.path, []).append(f)
    n_rep = n_true = tp = 0
    fp_lines, fn_detail = [], {}
    fp_good = []
    for path, bad in bad_by_file.items():
        fl = by_file.get(path, [])
        reported = {f.lineno for f in fl}
        bad_set = set(bad)
        tp += len(reported & bad_set)
        n_rep += len(reported)
        n_true += len(bad_set)
        for ln in sorted(reported - bad_set):
            fp_lines.append(f"{os.path.basename(path)}:{ln}")
        for ln in sorted(bad_set - reported):
            fn_detail[f"{os.path.basename(path)}:{ln}"] = bad[ln]
        good = good_by_file.get(path, set())
        for ln in sorted(reported & good):
            fp_good.append(f"{os.path.basename(path)}:{ln}")
    fn = n_true - tp
    prec = tp / max(1, n_rep)
    rec = tp / max(1, n_true)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "n_reported_lines": n_rep, "n_true_lines": n_true,
        "tp": tp, "fn": fn, "fp": len(fp_lines),
        "false_positive_lines": sorted(fp_lines),
        "false_negative_lines": sorted(fn_detail),
        "false_negatives": fn_detail,
        "false_positives_on_good": sorted(fp_good),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "n_findings": len(findings),
    }
