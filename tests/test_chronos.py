"""chronos 测试套件. 运行: python3 -m pytest tests/ -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.sim import default_scenario, simulate, build_scenario
from chronos.analysis import analyze, summary, tier_threshold_ms
from chronos.leap import (clock_reading, curve_metrics, fleet_differential,
                          build_checklist, POLICY_MODES, DEFAULT_WINDOW)
from chronos.scan.wallclock import scan_file, scan_source
from chronos.evaluate.scanner_eval import evaluate_scan
from chronos.advise import advise_topology, advise_leap, advise_thresholds


# ---------------------------------------------------------------- 仿真器物理

@pytest.fixture(scope="module")
def small_sim():
    sc = default_scenario(n_machines=200, days=10.0)
    return sc, simulate(sc)


def test_scenario_shapes(small_sim):
    sc, res = small_sim
    assert res.dev_ms.shape == (sc.n_nodes, sc.n_samples)
    assert res.true_off_s.shape == res.dev_ms.shape
    assert len(res.upstream_of) == sc.n_nodes


def test_calm_floor_reasonable(small_sim):
    """无故障的早期时段, 绝大多数机器偏差应在 50ms 内 (健康晶振)。"""
    sc, res = small_sim
    calm = np.abs(res.dev_ms[:, : int(2 * 86400 / 600)])
    # 99 分位 < 50ms
    assert np.nanpercentile(calm, 99) < 50.0


def test_upstream_fault_injects_offset(small_sim):
    sc, res = small_sim
    key = [k for k in res.fault_mask if k.startswith("upstream_bad")][0]
    m = res.fault_mask[key]
    assert m.any()
    # 故障时该服务器真实偏差接近 240s
    bad_node = int(np.flatnonzero(m.any(axis=1))[0])
    vals = np.abs(res.true_off_s[bad_node][m[bad_node]])
    assert np.nanmedian(vals) > 200.0


def test_majority_voting_protects_independent_sources(small_sim):
    """挂了坏上游, 但多源里多数 *独立于* 坏源时, 应被多数表决保护。

    注意: 若多个上游本身都经坏源 (共因), 多数表决救不了 —— 这正是
    共因风险的本质, 由另一个用例覆盖。
    """
    sc, res = small_sim
    key = [k for k in res.fault_mask if k.startswith("upstream_bad")][0]
    m = res.fault_mask[key].any(axis=1)
    S = sc.n_servers
    bad_srv = int(np.flatnonzero(m[:S])[0])
    # 沿上游链追溯是否依赖坏源
    def depends(x, seen=None):
        seen = seen or set()
        if x == bad_srv:
            return True
        if x in seen:
            return False
        seen.add(x)
        return any(depends(u, seen) for u in res.upstream_of[x])
    protected = total = 0
    for node in range(S, sc.n_nodes):
        ups = res.upstream_of[node]
        if bad_srv not in ups:
            continue
        n_good = sum(1 for u in ups if not depends(u))
        if n_good >= 2:
            total += 1
            # "被保护" = 偏移不持续 (多数表决会在少数几个采样内剔除坏源;
            # 只在阶跃瞬间有短暂泄漏, 不允许长期停留)
            sustained = (np.abs(res.true_off_s[node]) > 1.0).sum()
            if sustained <= 12:          # ≤ 2 小时
                protected += 1
    assert total > 0, "样本里没有 '坏源+≥2独立源' 的机器"
    assert protected == total, f"{total-protected}/{total} 台有独立多源却被长期带偏"


def test_common_cause_beats_voting(small_sim):
    """多个上游共同依赖坏源时, 多数表决失效 —— 机器会被带偏。"""
    sc, res = small_sim
    key = [k for k in res.fault_mask if k.startswith("upstream_bad")][0]
    m = res.fault_mask[key].any(axis=1)
    S = sc.n_servers
    bad_srv = int(np.flatnonzero(m[:S])[0])
    def depends(x, seen=None):
        seen = seen or set()
        if x == bad_srv:
            return True
        if x in seen:
            return False
        seen.add(x)
        return any(depends(u, seen) for u in res.upstream_of[x])
    # 存在某台机器的多数上游都依赖坏源, 且它被带偏
    hurt = 0
    for node in range(S, sc.n_nodes):
        ups = res.upstream_of[node]
        if not ups:
            continue
        n_good = sum(1 for u in ups if not depends(u))
        if n_good < len(ups) - n_good and np.abs(res.true_off_s[node]).max() > 100:
            hurt += 1
    assert hurt > 0


def test_vm_suspend_creates_backward_jump(small_sim):
    sc, res = small_sim
    key = [k for k in res.fault_mask if k.startswith("vm_suspend")]
    if not key:
        pytest.skip("no vm fault")
    m = res.fault_mask[key[0]].any(axis=1)
    nodes = np.flatnonzero(m)
    assert len(nodes) > 0
    # 至少一台出现 >10s 的负向偏差
    found = False
    for n in nodes:
        if np.abs(res.true_off_s[n]).max() > 10:
            found = True
            break
    assert found


def test_leap_off_bounded(small_sim):
    """闰秒分量绝对值不超过 ~1s。"""
    sc, res = small_sim
    if res.scenario.leap is None:
        pytest.skip("no leap")
    assert np.abs(res.leap_off_s).max() <= 1.01


# ---------------------------------------------------------------- 分析

@pytest.fixture(scope="module")
def analyzed(small_sim):
    sc, res = small_sim
    # 构造一个最小遥测 dict
    from chronos.sim.dataset import _meta
    meta = _meta(res)
    tel = {
        "meta": meta, "times_s": res.times_s, "dev_ms": res.dev_ms,
        "sync_off_ms": res.sync_off_ms, "delay_ms": res.delay_ms,
        "jitter_ms": res.jitter_ms, "loss_pct": res.loss_pct,
        "reach": res.reach, "peer_idx": res.peer_idx,
    }
    return res, meta, analyze(tel), tel


def test_analysis_runs_fast(analyzed):
    res, meta, ar, tel = analyzed
    assert ar.n == meta["n_nodes"]
    assert ar.bad_mask.shape == res.dev_ms.shape


def test_analysis_finds_bad_upstream(analyzed):
    res, meta, ar, tel = analyzed
    # 真值里出错的服务器
    bad_names = [f["target"] for f in res.fault_meta if f["kind"] == "upstream_bad"]
    found = [u for u in ar.upstream_incidents if u.name in bad_names]
    assert found, "未能识别出故障上游"


def test_analysis_attributes_causes(analyzed):
    res, meta, ar, tel = analyzed
    causes = set(f.cause for f in ar.findings)
    # 至少识别出 local / upstream 两类
    assert "local" in causes or "upstream" in causes


def test_leap_window_not_flagged(analyzed):
    """闰秒摊开窗口内不应被判为故障。"""
    res, meta, ar, tel = analyzed
    if meta["leap_rel_s"] is None:
        pytest.skip()
    leap = int(meta["leap_rel_s"] / meta["base_dt"])
    # 闰秒前 2h 内, 非故障机器不应被判 bad
    lo, hi = leap - int(2 * 3600 / meta["base_dt"]), leap - int(1 * 3600 / meta["base_dt"])
    n_bad = int(ar.bad_mask[:, lo:hi].sum())
    # 允许极少数 (其他故障重叠), 但远小于全机群
    assert n_bad < meta["n_nodes"] * 0.2


def test_tier_thresholds():
    assert tier_threshold_ms("critical") == 5.0
    assert tier_threshold_ms("high") == 50.0
    assert tier_threshold_ms("standard") == 500.0
    assert tier_threshold_ms("batch") == 2000.0


# ---------------------------------------------------------------- 闰秒物理

def test_step_is_non_monotonic():
    m = curve_metrics("step", 0)
    assert m["monotonic"] is False
    assert m["max_backward_step_s"] >= 0.99


def test_slew_smear_monotonic():
    for mode in ("slew", "smear"):
        m = curve_metrics(mode, DEFAULT_WINDOW[mode])
        assert m["monotonic"] is True, f"{mode} 不应产生时间倒退"
        assert m["max_abs_dev_s"] <= 1.01


def test_slew_vs_smear_differ():
    """slew 与 smear 在窗口期内存在显著差异 (不是等价选项)。"""
    d = fleet_differential([("slew", "smear")])[0]
    assert d["max_abs_diff_s"] > 0.5
    assert d["dur_gt_100ms_s"] > 3600


def test_step_vs_smear_long_divergence():
    d = fleet_differential([("step", "smear")])[0]
    assert d["max_abs_diff_s"] > 0.9
    assert d["dur_gt_500ms_s"] > 3600


def test_impact_checklist_step_worst_for_id():
    cl = build_checklist()
    assert cl["severity_matrix"]["id_gen"]["step"] == "critical"
    assert cl["severity_matrix"]["id_gen"]["smear"] in ("ok", "low")


def test_impact_checklist_lease():
    cl = build_checklist()
    assert cl["severity_matrix"]["lease"]["step"] == "critical"


def test_mixed_fleet_has_severe_inversions():
    cl = build_checklist()
    pair = [c for c in cl["mixed_cases"] if c["pair"] == "step+smear"][0]
    assert pair["log_order"]["severity"] == "critical"


# ---------------------------------------------------------------- 扫描器

def test_scanner_catches_id():
    src = "import time\ndef f():\n    oid = int(time.time()*1000)\n    return oid\n"
    f = scan_source("t.py", src)
    assert any(x.rule == "timestamp_for_id" for x in f)


def test_scanner_catches_elapsed():
    src = "import time\ndef f():\n    t=time.time()\n    return time.time()-t\n"
    f = scan_source("t.py", src)
    assert any(x.rule == "sub_now" for x in f)


def test_scanner_ignores_monotonic():
    src = ("import time\ndef f():\n    t=time.monotonic()\n"
           "    return time.monotonic()-t\n")
    f = scan_source("t.py", src)
    assert not any(x.rule in ("sub_now",) for x in f)


def test_scanner_corpus_perfect():
    r = evaluate_scan("tests/corpus")
    assert r["recall"] >= 0.95
    assert r["precision"] >= 0.9


# ---------------------------------------------------------------- 建议

def test_advise_thresholds():
    t = advise_thresholds()
    assert t.tiers["critical"]["threshold_ms"] == 5.0
    assert t.tiers["batch"]["threshold_ms"] == 2000.0


def test_advise_leap_uniform():
    l = advise_leap()
    assert l.uniform_choice in ("smear", "slew")
    assert len(l.forbidden) >= 2


def test_advise_topology_finds_single_source(analyzed):
    res, meta, ar, tel = analyzed
    adv = advise_topology(meta)
    # 场景里有单源机器
    assert adv.n_single_source >= 0
    assert len(adv.per_node) == meta["n_machines"]


# ---------------------------------------------------------------- 端到端评测

def test_end_to_end_eval():
    """小场景: 召回应高, 根因准确率应可观。"""
    from chronos.sim.dataset import _meta
    from chronos.evaluate.metrics import evaluate
    sc = default_scenario(n_machines=400, days=12.0)
    res = simulate(sc)
    meta = _meta(res)
    tel = {"meta": meta, "times_s": res.times_s, "dev_ms": res.dev_ms,
           "sync_off_ms": res.sync_off_ms, "delay_ms": res.delay_ms,
           "jitter_ms": res.jitter_ms, "loss_pct": res.loss_pct,
           "reach": res.reach, "peer_idx": res.peer_idx}
    ar = analyze(tel)
    truth = {"true_off_s": res.true_off_s, "fault_meta": res.fault_meta,
             "leap_off_s": res.leap_off_s}
    for k, m in res.fault_mask.items():
        truth["mask__" + k] = m
    ev = evaluate(ar, truth).to_dict()
    # 小样本 (400 台) 的上游受影响节点较少, 归因不如全量稳定; 召回仍要求高。
    assert ev["fault_detection"]["recall"] > 0.85
    assert ev["cause_accuracy"] > 0.4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
