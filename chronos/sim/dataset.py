"""把仿真结果落盘为 "可观测遥测包" + "隐藏真值包" 两个文件.

分析模块只会 load telemetry; 评测模块再额外 load truth。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np

from ..util import UTC
from .clocksim import SimResult, simulate
from .scenario import Scenario, default_scenario


def _meta(res: SimResult) -> dict:
    sc = res.scenario
    return {
        "scenario": sc.name, "seed": sc.seed, "start": sc.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "days": sc.days, "base_dt": sc.base_dt, "n_samples": int(res.T),
        "n_nodes": int(res.N), "n_servers": int(sc.n_servers),
        "n_machines": int(len(sc.machines)),
        "node_names": res.node_names,
        "node_is_server": res.node_is_server.tolist(),
        "node_stratum": res.node_stratum.tolist(),
        "node_tier": res.node_tier,
        "node_region": res.node_region,
        "upstream_of": res.upstream_of,
        "leap_when": (sc.leap.when.strftime("%Y-%m-%dT%H:%M:%SZ") if sc.leap else None),
        "leap_rel_s": ((sc.leap.when - sc.start).total_seconds() if sc.leap else None),
        "fault_meta": res.fault_meta,
    }


def save(res: SimResult, out_dir: str, tag: str = "default") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    meta = _meta(res)
    # ---- 可观测遥测
    tel = {
        "meta": meta,
        "times_s": res.times_s.tolist(),
        "dev_ms": res.dev_ms,
        "sync_off_ms": res.sync_off_ms,
        "delay_ms": res.delay_ms,
        "jitter_ms": res.jitter_ms,
        "loss_pct": res.loss_pct,
        "reach": res.reach,
        "peer_idx": res.peer_idx,
    }
    tel_path = os.path.join(out_dir, f"{tag}.telemetry.npz")
    arrs = {k: v for k, v in tel.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(tel_path, **arrs)
    with open(os.path.join(out_dir, f"{tag}.telemetry.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "times_s": tel["times_s"]}, fh, ensure_ascii=False)

    # ---- 真值 (评测专用)
    truth = {
        "true_off_s": res.true_off_s,
        "leap_off_s": res.leap_off_s,
        "fault_meta": json.dumps(res.fault_meta, ensure_ascii=False),
    }
    for k, m in res.fault_mask.items():
        truth["mask__" + k.replace("@", "_")] = m
    np.savez_compressed(os.path.join(out_dir, f"{tag}.truth.npz"), **truth)
    with open(os.path.join(out_dir, f"{tag}.truth.json"), "w", encoding="utf-8") as fh:
        json.dump({"fault_meta": res.fault_meta,
                   "scenario": sc_public(res.scenario)}, fh, ensure_ascii=False)
    return {"telemetry": tel_path, "tag": tag}


def sc_public(sc: Scenario) -> dict:
    d = sc.to_dict()
    # 真值里保留完整拓扑 (含上游关系与策略), 评测/建议模块需要
    return d


def load_telemetry(path: str) -> dict:
    """读取遥测包。path 可为 .npz 或前缀。"""
    base = path[:-4] if path.endswith(".npz") else path
    meta_path = base.replace(".telemetry", "") + ".telemetry.json"
    data = dict(np.load(base + ".telemetry.npz" if not base.endswith(".telemetry") else base + ".npz"))
    with open(meta_path, encoding="utf-8") as fh:
        j = json.load(fh)
    data["meta"] = j["meta"]
    data["times_s"] = np.asarray(j["times_s"], dtype=np.float64)
    return data


def load_truth(path: str) -> dict:
    base = path[:-4] if path.endswith(".npz") else path
    base = base.replace(".telemetry", "")
    data = dict(np.load(base + ".truth.npz"))
    with open(base + ".truth.json", encoding="utf-8") as fh:
        j = json.load(fh)
    data["fault_meta"] = j["fault_meta"]
    data["scenario"] = j["scenario"]
    return data


def build_and_save(out_dir: str = "data", tag: str = "fleet", **kw):
    sc = default_scenario(**kw)
    res = simulate(sc)
    paths = save(res, out_dir, tag)
    return sc, res, paths
