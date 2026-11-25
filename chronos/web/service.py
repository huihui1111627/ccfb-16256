"""数据服务层: 加载遥测、跑分析/评测/建议/闰秒, 结果内存缓存.

页面只通过这里取数, 保持前端简单。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..analysis.attribution import AnalysisResult, analyze, summary
from ..advise.advisor import build_advice
from ..evaluate.metrics import evaluate
from ..evaluate.scanner_eval import evaluate_scan
from ..leap.impact import build_checklist
from ..leap.policies import build_curves, curve_metrics, mixed_fleet_summary, DEFAULT_WINDOW, MODE_LABEL
from ..scan.wallclock import scan_dir, summarize as scan_summarize
from ..sim.dataset import load_telemetry, load_truth
from ..util import fmt_utc, parse_utc

DEFAULT_DATA = "data/fleet"


class Service:
    def __init__(self, data_prefix: str = DEFAULT_DATA, corpus_dir: str = "tests/corpus"):
        self.data_prefix = data_prefix
        self.corpus_dir = corpus_dir
        self._tel = None
        self._truth = None
        self._ar: Optional[AnalysisResult] = None
        self._eval = None
        self._scan = None
        self._leap = None

    # ---------------- lazy
    @property
    def tel(self) -> dict:
        if self._tel is None:
            self._tel = load_telemetry(self.data_prefix + ".telemetry.npz")
        return self._tel

    @property
    def truth(self) -> dict:
        if self._truth is None:
            self._truth = load_truth(self.data_prefix + ".telemetry.npz")
        return self._truth

    @property
    def meta(self) -> dict:
        return self.tel["meta"]

    @property
    def ar(self) -> AnalysisResult:
        if self._ar is None:
            self._ar = analyze(self.tel)
        return self._ar

    @property
    def eval(self) -> dict:
        if self._eval is None:
            self._eval = evaluate(self.ar, self.truth).to_dict()
            self._eval["detail"] = evaluate(self.ar, self.truth).detail
        return self._eval

    @property
    def scan(self) -> dict:
        if self._scan is None:
            findings = scan_dir(self.corpus_dir)
            self._scan = scan_summarize(findings)
            self._scan["evaluation"] = evaluate_scan(self.corpus_dir)
        return self._scan

    @property
    def leap(self) -> dict:
        if self._leap is None:
            self._leap = build_checklist()
        return self._leap

    # ---------------- 概览
    def overview(self) -> dict:
        m = self.meta
        ar = self.ar
        s = summary(ar)
        return {
            "scenario": m["scenario"], "start": m["start"], "days": m["days"],
            "n_nodes": m["n_nodes"], "n_servers": m["n_servers"], "n_machines": m["n_machines"],
            "base_dt": m["base_dt"], "n_samples": m["n_samples"],
            "leap_when": m["leap_when"], "leap_rel_s": m["leap_rel_s"],
            "causes": s["causes"], "n_bad": s["n_bad"],
            "n_upstream_incidents": len(ar.upstream_incidents),
            "n_network_incidents": len(ar.network_incidents),
            "eval": self.eval,
        }

    # ---------------- 偏差热力图 (按行=节点, 列=时间降采样)
    def heatmap(self, max_rows: int = 900, bins: int = 420) -> dict:
        dev = np.abs(np.asarray(self.ar.dev_ms))
        N, T = dev.shape
        # 时间分箱取每箱最大
        edges = np.linspace(0, T, bins + 1).astype(int)
        reb = np.zeros((N, bins))
        for k in range(bins):
            a, b = edges[k], max(edges[k] + 1, edges[k + 1])
            reb[:, k] = dev[:, a:b].max(axis=1)
        # 节点按 tier 再按中位偏差排序, 让同类聚在一起
        meta = self.meta
        tier_rank = {"server": -1, "critical": 0, "high": 1, "standard": 2, "batch": 3}
        order = sorted(range(N), key=lambda i: (tier_rank.get(meta["node_tier"][i], 9),
                                                float(np.nanmedian(dev[i]))))
        # 采样行
        rows = order[:max_rows] if N <= max_rows else order[:: max(1, N // max_rows)][:max_rows]
        mat = reb[rows]
        # 对数压缩到 0..1 (ms): 0.1ms -> 0, 300s -> 1
        v = np.clip(np.log10(np.maximum(mat, 0.1)) / np.log10(3e5), 0, 1)
        # 每行标签
        labels = [meta["node_names"][i] for i in rows]
        tiers = [meta["node_tier"][i] for i in rows]
        return {
            "data": np.round(v, 3).tolist(),
            "labels": labels, "tiers": tiers, "idx": [int(i) for i in rows],
            "times": [float(self.tel["times_s"][edges[k]]) for k in range(bins)],
            "n_rows_total": N, "n_rows_shown": len(rows),
            "vmin_ms": 0.1, "vmax_ms": 300000.0,
            "leap_rel_s": self.meta["leap_rel_s"],
        }

    # ---------------- 分布 (按 tier + 时间)
    def distribution(self, bins: int = 96) -> dict:
        dev = np.abs(np.asarray(self.ar.dev_ms))
        meta = self.meta
        tiers = meta["node_tier"]
        T = dev.shape[1]
        edges = np.linspace(0, T, bins + 1).astype(int)
        out = {}
        for tier in ["critical", "high", "standard", "batch"]:
            idx = [i for i in range(len(tiers)) if tiers[i] == tier]
            if not idx:
                continue
            p50, p95, p99 = [], [], []
            for k in range(bins):
                a, b = edges[k], max(edges[k] + 1, edges[k + 1])
                seg = dev[np.ix_(idx, range(a, b))]
                p50.append(float(np.nanpercentile(seg, 50)))
                p95.append(float(np.nanpercentile(seg, 95)))
                p99.append(float(np.nanpercentile(seg, 99)))
            out[tier] = {
                "p50": p50, "p95": p95, "p99": p99,
                "threshold_ms": {"critical": 5, "high": 50, "standard": 500, "batch": 2000}[tier],
                "n": len(idx),
            }
        times = [float(self.tel["times_s"][edges[k]]) for k in range(bins)]
        return {"tiers": out, "times": times, "leap_rel_s": meta["leap_rel_s"]}

    # ---------------- 服务器层级与健康
    def hierarchy(self) -> dict:
        meta = self.meta
        ar = self.ar
        names = meta["node_names"]
        upstream_of = [list(u) for u in meta["upstream_of"]]
        N = len(names)
        nodes = []
        for i in range(N):
            if not meta["node_is_server"][i]:
                continue
            f = ar.findings[i]
            nodes.append({
                "idx": i, "name": names[i], "stratum": int(meta["node_stratum"][i]),
                "region": meta["node_region"][i],
                "upstream": upstream_of[i],
                "cause": f.cause, "peak_ms": round(f.peak_ms, 2),
                "baseline_ms": round(float(ar.baseline_ms[i]), 3),
                "sigma_ms": round(float(ar.sigma_ms[i]), 3),
                "n_children": sum(1 for k in range(N) if i in upstream_of[k]),
                "culprit_of_incident": any(u.culprit == i for u in ar.upstream_incidents),
            })
        return {"nodes": nodes, "n_servers": len(nodes)}

    def incidents(self) -> dict:
        ar = self.ar
        from datetime import timedelta
        start = parse_utc(self.meta["start"])
        ups = []
        for u in ar.upstream_incidents:
            ups.append({
                "culprit": u.culprit, "name": u.name,
                "onset": fmt_utc(start + timedelta(seconds=u.onset_s)),
                "peak_ms": round(u.peak_ms, 1), "direction": u.direction,
                "direct_children": u.direct_children,
                "affected": len(u.affected), "protected": u.protected,
                "confidence": round(u.confidence, 2),
                "evidence": u.evidence,
                "affected_names": [ar.node_names[k] for k in u.affected[:60]],
            })
        nets = []
        for n in ar.network_incidents:
            nets.append({
                "region": n.key,
                "onset": fmt_utc(start + timedelta(seconds=n.onset_s)),
                "end": fmt_utc(start + timedelta(seconds=n.end_s)),
                "affected": len(n.affected),
                "delay_ratio": round(n.delay_ratio, 2),
                "jitter_ratio": round(n.jitter_ratio, 2),
                "loss_add": round(n.loss_add, 1),
                "evidence": n.evidence,
            })
        return {"upstream": ups, "network": nets}

    # ---------------- 单节点下钻
    def node_detail(self, idx: int) -> dict:
        ar = self.ar
        meta = self.meta
        if idx < 0 or idx >= len(ar.findings):
            return {"error": "bad idx"}
        f = ar.findings[idx]
        dev = np.asarray(self.ar.dev_ms[idx])
        delay = np.asarray(self.tel["delay_ms"][idx])
        jitter = np.asarray(self.tel["jitter_ms"][idx])
        loss = np.asarray(self.tel["loss_pct"][idx])
        # 降采样
        step = max(1, len(dev) // 600)
        return {
            "name": f.name, "tier": f.tier, "region": f.region, "cause": f.cause,
            "subtype": f.subtype, "peak_ms": round(f.peak_ms, 2), "rms_ms": round(f.rms_ms, 2),
            "confidence": round(f.confidence, 2), "evidence": f.evidence,
            "culprit": f.culprit,
            "culprit_name": ar.node_names[f.culprit] if f.culprit is not None else None,
            "onset": f.onset_s, "clear": f.clear_s,
            "threshold_ms": round(float(ar.threshold_ms[idx]), 2),
            "baseline_ms": round(float(ar.baseline_ms[idx]), 3),
            "upstream": [meta["node_names"][u] for u in meta["upstream_of"][idx]],
            "times": [float(self.tel["times_s"][i]) for i in range(0, len(dev), step)],
            "dev_ms": [round(float(x), 2) for x in dev[::step]],
            "delay_ms": [None if np.isnan(x) else round(float(x), 2) for x in delay[::step]],
            "jitter_ms": [None if np.isnan(x) else round(float(x), 3) for x in jitter[::step]],
            "loss_pct": [round(float(x), 1) for x in loss[::step]],
            "leap_rel_s": meta["leap_rel_s"],
        }

    def node_list(self, cause: str = None, limit: int = 200, include_ok: bool = False) -> dict:
        ar = self.ar
        out = []
        for f in ar.findings:
            if cause and f.cause != cause:
                continue
            if f.cause == "ok" and not include_ok:
                continue
            out.append({
                "idx": f.idx, "name": f.name, "tier": f.tier, "region": f.region,
                "cause": f.cause, "subtype": f.subtype, "peak_ms": round(f.peak_ms, 2),
                "onset_s": f.onset_s, "culprit": f.culprit, "confidence": round(f.confidence, 2),
            })
        out.sort(key=lambda x: -x["peak_ms"])
        return {"total": len(out), "nodes": out[:limit]}

    # ---------------- 闰秒
    def leap_data(self, span_h: float = 52.0) -> dict:
        cs = build_curves(span_h=span_h)
        metrics = {m: curve_metrics(m, DEFAULT_WINDOW[m]) for m in MODE_LABEL}
        mixed = mixed_fleet_summary()
        cl = self.leap
        step = max(1, len(cs.tau) // 1200)
        return {
            "tau_min_h": round(float(cs.tau[0]) / 3600, 2),
            "times_h": [round(float(x) / 3600, 4) for x in cs.tau[::step]],
            "curves": {m: [round(float(x) * 1000, 2) for x in cs.devs[m][::step]] for m in cs.devs},
            "readings": {m: [round(float(x - cs.tau[0]), 3) for x in cs.readings[m][::step]] for m in cs.readings},
            "labels": cs.labels,
            "metrics": metrics,
            "mixed": mixed,
            "checklist": cl,
        }

    # ---------------- 扫描 / 建议
    def scan_data(self) -> dict:
        return self.scan

    def advice(self) -> dict:
        return build_advice(self.meta, self.leap)


SERVICE = Service()
