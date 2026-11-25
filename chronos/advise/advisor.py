"""对时配置建议.

三部分:
1. **上游编排**: 每台机器应跟哪几台上游。原则:
     - 至少 3 个真正独立的上游 (才能让多数表决剔除单点错误);
     - 散在不同 tier-1 源上, 避免共因 (同一上游挂掉/走偏时不全军覆没);
     - 优先同区域低时延的上游, 但至少 1 个跨区做冗余;
     - 禁止单源 (见规划器会重排单源节点)。
   输出每台机器当前配置的问题 + 建议的上游集合 + 预期效果。

2. **闰秒策略**: 全机群统一 (step / slew / smear 三选一), 结合业务。
   给出各方案对每类业务的代价, 以及 "禁止混用" 的具体原因。

3. **分级阈值**: 按业务对时间的敏感度给告警阈值 (critical/high/standard/batch),
   并给出配套的观测窗口与 "是否纳入巡检强约束" 建议。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from ..sim.scenario import BUSINESS_TIERS


# ---------------------------------------------------------------- 拓扑建议

@dataclass
class UpstreamAdvice:
    node: int
    name: str
    current: List[int]
    recommended: List[int]
    issues: List[str]
    severity: str        # ok | warn | critical
    reason: str


@dataclass
class TopologyAdvice:
    per_node: List[UpstreamAdvice]
    n_single_source: int
    n_common_mode: int
    n_insufficient_diversity: int
    summary: str


def _distinct_tier1(node_idx: int, upstream_of: List[List[int]],
                    tier1: Set[int], parents: Dict[int, List[int]]) -> Set[int]:
    """节点最终依赖到哪些 tier-1 源 (沿上游链追溯到 stratum-1)。"""
    out: Set[int] = set()
    stack = list(upstream_of[node_idx])
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in tier1:
            out.add(x)
        for p in parents.get(x, []):
            stack.append(p)
    return out


def advise_topology(meta: dict, max_recommend: int = 4) -> TopologyAdvice:
    """基于遥测 meta (拓扑) 给每台机器的上游配置建议。"""
    names = meta["node_names"]
    upstream_of = [list(u) for u in meta["upstream_of"]]
    is_server = np.asarray(meta["node_is_server"], dtype=bool)
    stratum = np.asarray(meta["node_stratum"])
    region = meta["node_region"]
    tier = meta["node_tier"]
    N = len(names)

    tier1 = set(int(i) for i in range(N) if stratum[i] == 1)
    parents = {i: upstream_of[i] for i in range(N) if upstream_of[i]}
    # 每个区域的 tier-2 中继
    relays_by_region: Dict[str, List[int]] = {}
    for i in range(N):
        if stratum[i] == 2:
            relays_by_region.setdefault(region[i], []).append(i)

    # 全局所有可用的 stratum-2 (用于跨区冗余推荐)
    all_relays = [i for i in range(N) if stratum[i] == 2]

    per_node: List[UpstreamAdvice] = []
    n_single = n_common = n_insufficient = 0
    for i in range(N):
        if is_server[i]:
            continue
        cur = upstream_of[i]
        issues: List[str] = []
        sev = "ok"
        # 1) 单源
        if len(cur) <= 1:
            issues.append(f"只有 {len(cur)} 个上游, 无法做多数表决")
            sev = "critical"
            n_single += 1
        # 2) 追溯 tier-1 多样性
        t1 = _distinct_tier1(i, upstream_of, tier1, parents)
        if len(cur) >= 2 and len(t1) <= 1:
            issues.append(f"上游虽多但最终只依赖 {len(t1)} 个 tier-1 源, 存在共因")
            sev = "critical"
            n_common += 1
        elif len(cur) < 2 and len(t1) <= 1 and cur:
            n_insufficient += 1
        # 3) 业务敏感度: critical 业务至少要 3 源 / 2 个 tier-1
        sens = BUSINESS_TIERS.get(tier[i], (1.0, ""))[0]
        need_n = 3 if sens <= 0.05 else 2
        if len(cur) < need_n and sev == "ok":
            issues.append(f"{tier[i]} 业务建议 ≥{need_n} 源, 当前 {len(cur)}")
            sev = "warn"

        # 建议上游: 本区 2 台中继 + 1 台异地 + (关键业务) 1 台 tier1
        local = [r for r in relays_by_region.get(region[i], []) if r not in cur] or \
                [r for r in relays_by_region.get(region[i], [])]
        rec: List[int] = []
        rec.extend(local[:2])
        # 找一台不同 tier-1 链的异地中继
        local_t1 = set().union(*[_distinct_tier1(r, upstream_of, tier1, parents) for r in rec]) if rec else set()
        for r in all_relays:
            if r in rec:
                continue
            rt1 = _distinct_tier1(r, upstream_of, tier1, parents)
            if rt1 - local_t1:
                rec.append(r)
                local_t1 |= rt1
            if len(rec) >= max_recommend:
                break
        if sens <= 0.05 and len(rec) < max_recommend:
            for t in sorted(tier1):
                if t not in rec and len(rec) < max_recommend:
                    rec.append(t)
        rec = rec[:max_recommend]
        reason = "本区低时延中继 + 跨区冗余 + 独立 tier-1 源, 满足多数表决和多源冗余"
        per_node.append(UpstreamAdvice(
            node=i, name=names[i], current=cur, recommended=sorted(set(rec)),
            issues=issues, severity=sev,
            reason=reason if sev != "ok" else "配置已满足冗余要求"))

    summary = (f"共 {len(per_node)} 台业务机: {n_single} 台单源 (无法表决), "
               f"{n_common} 台多源但共因 (最终只依赖 1 个 tier-1), "
               f"{n_insufficient} 台 tier-1 多样性不足")
    return TopologyAdvice(per_node=per_node, n_single_source=n_single,
                          n_common_mode=n_common,
                          n_insufficient_diversity=n_insufficient,
                          summary=summary)


# ---------------------------------------------------------------- 闰秒策略

@dataclass
class LeapAdvice:
    uniform_choice: str
    rationale: str
    per_business_tier: Dict[str, dict]
    forbidden: List[str]
    migration: List[str]


def advise_leap(leap_result: dict = None) -> LeapAdvice:
    """给全机群统一的闰秒策略。leap_result 来自 leap.build_checklist()。"""
    if leap_result is None:
        from ..leap.impact import build_checklist
        leap_result = build_checklist()
    matrix = leap_result["severity_matrix"]
    # 选 "最坏原语严重度最低" 的方案; step 永远因时间倒退淘汰
    def worst_sev(m):
        return max({"ok": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[matrix[p][m]]
                   for p in matrix)
    candidates = ["slew", "smear", "none"]   # step 直接排除
    best = min(candidates, key=lambda m: (worst_sev(m), 0 if m == "smear" else 1))
    per_tier = {}
    for tier, (thr, desc) in BUSINESS_TIERS.items():
        if tier in ("critical", "high"):
            per_tier[tier] = {
                "threshold_ms": thr * 1000,
                "recommended": "统一 smear, 对账/计费子集用 slew 1h 并与外部 UTC 强对齐",
                "why": desc,
            }
        else:
            per_tier[tier] = {
                "threshold_ms": thr * 1000,
                "recommended": "统一 smear 即可",
                "why": desc,
            }
    forbidden = [
        "严禁 step 与 slew/smear 混用: 抹平窗口内两组相差可达 1s, 租约/日志/编号全部错乱",
        "严禁任何机器配置 '忽略闰秒(none)': 会与已纠正机器永久差 1s",
        "严禁部分机器 step: 时间倒退 1s 直接破坏单调时钟与顺序假设",
    ]
    migration = [
        "阶段1: 冻结变更, 全量盘点每台的 leap 配置 (ntpd/chrony/kernel) 与是否 -x 禁 step",
        "阶段2: 全机群统一改为 smear (24h), 除非该组需与外部 UTC 对齐则统一 slew 1h",
        "阶段3: 闰秒前 24h 做一次全量对时校准, 确认无机器仍在 step/none",
        "阶段4: 闰秒窗口 (±1h) 内对 critical 业务降级跨机强依赖 (锁改租约上限/幂等)",
    ]
    return LeapAdvice(
        uniform_choice=best,
        rationale=("step 会造成时间倒退 (租约提前失效 + 编号回退 + 日志逆序); "
                   "slew/smear 单调无倒退, 是唯一安全方向。全机群必须同一种抹平窗口, "
                   "否则不同组在窗口内长期相差最多 1s"),
        per_business_tier=per_tier, forbidden=forbidden, migration=migration)


# ---------------------------------------------------------------- 分级阈值

@dataclass
class ThresholdAdvice:
    tiers: Dict[str, dict]
    rationale: str
    window_s: int


def advise_thresholds() -> ThresholdAdvice:
    tiers = {}
    for t, (thr, desc) in BUSINESS_TIERS.items():
        tiers[t] = {
            "threshold_ms": thr * 1000,
            "window_s": 600 if t in ("critical", "high") else 1800,
            "action": "立即告警+自动摘除上游" if t == "critical"
                      else ("告警+根因定位" if t == "high" else "工单巡检"),
            "why": desc,
        }
    return ThresholdAdvice(
        tiers=tiers,
        rationale=("阈值随业务对时间的敏感度收紧。critical 业务 (交易/订单/锁) 取 5ms, "
                   "high (支付/风控) 50ms, standard 500ms, batch 2s。同时要求至少 6σ "
                   "高于该机自身波动, 避免正常抖动刷屏。时间服务器本身统一收紧到 10ms。"),
        window_s=600)


# ---------------------------------------------------------------- 汇总

def build_advice(meta: dict, leap_result: dict = None) -> dict:
    topo = advise_topology(meta)
    leap = advise_leap(leap_result)
    thr = advise_thresholds()
    # 只返回有问题的节点建议 + 统计, 避免几千条把页面撑爆
    problem = [a for a in topo.per_node if a.severity != "ok"]
    return {
        "topology": {
            "summary": topo.summary,
            "n_single_source": topo.n_single_source,
            "n_common_mode": topo.n_common_mode,
            "n_insufficient_diversity": topo.n_insufficient_diversity,
            "n_problem": len(problem),
            "examples": [{
                "name": a.name, "current": a.current, "recommended": a.recommended,
                "issues": a.issues, "severity": a.severity,
            } for a in problem[:40]],
        },
        "leap": {
            "uniform_choice": leap.uniform_choice,
            "rationale": leap.rationale,
            "per_business_tier": leap.per_business_tier,
            "forbidden": leap.forbidden,
            "migration": leap.migration,
        },
        "thresholds": {
            "tiers": thr.tiers, "rationale": thr.rationale, "window_s": thr.window_s,
        },
    }
