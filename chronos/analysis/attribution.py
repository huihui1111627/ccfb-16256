"""偏差检测、分级告警、根因归因、上游故障定位.

输入只有遥测 (dev_ms / delay / jitter / loss / reach / peer 与拓扑),
不读取真值。输出结构化结论, 供页面和评测使用。

关键设计
--------
* **闰秒摊开不是故障**: smear 策略会让全机群在每个闰秒前后 ±13h 内出现
  最大 500ms 的 "计划内" 偏差。检测时按遥测里的 `leap_rel_s` 把该窗口
  标记为 `scheduled`, 既不报警也不参与基线, 避免把策略效应误判成故障。
* **分级阈值**: 按业务 tier 取 (critical 5ms / high 50ms / standard 500ms
  / batch 2000ms), 同时要求至少 6σ 高于该机自身波动, 防止敏感业务被
  正常抖动刷屏。时间服务器统一收紧到 10ms。
* **归因**: 对每台越限机器, 依次检查
    1) 网络证据 —— 偏差窗口与路径时延/抖动/丢包恶化重叠;
    2) 上游共因 —— 与它共享某上游的兄弟在同一时段同向、同幅偏移;
    3) 本机独有 —— 兄弟正常而只有它偏 (ntpd 停摆 / 人工改时 / 晶振坏)。
  上游服务器自身是否也偏, 用于在 "上游错" 与 "本机错" 之间做区分。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..sim.scenario import BUSINESS_TIERS
from ..util import episodes, fmt_utc


@dataclass
class NodeFinding:
    idx: int
    name: str
    tier: str
    region: str
    cause: str                 # local | upstream | network | scheduled | ok
    peak_ms: float
    rms_ms: float
    onset_s: Optional[float]
    clear_s: Optional[float]
    culprit: Optional[int]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    windows: List[Tuple[float, float]] = field(default_factory=list)
    subtype: str = ""          # vm_pause | manual_step | dead_ntpd | bad_osc | path | feed | ""

    @property
    def peak_s(self) -> float:
        return self.peak_ms / 1000.0


@dataclass
class UpstreamIncident:
    culprit: int
    name: str
    onset_s: float
    detected_s: float
    peak_ms: float
    direction: float
    direct_children: int
    affected: List[int]
    protected: int
    confidence: float
    evidence: List[str] = field(default_factory=list)

    @property
    def detection_latency_s(self) -> float:
        return max(0.0, self.detected_s - self.onset_s)


@dataclass
class NetworkIncident:
    key: str
    onset_s: float
    end_s: float
    affected: List[int]
    delay_ratio: float
    jitter_ratio: float
    loss_add: float
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    threshold_ms: np.ndarray
    dev_ms: np.ndarray
    bad_mask: np.ndarray
    findings: List[NodeFinding]
    upstream_incidents: List[UpstreamIncident]
    network_incidents: List[NetworkIncident]
    baseline_ms: np.ndarray
    sigma_ms: np.ndarray
    node_names: List[str]
    tier: List[str]
    region: List[str]
    times_s: np.ndarray
    meta: dict
    scheduled_mask: np.ndarray = None   # [N,T] 闰秒计划窗口

    def bad_nodes_at(self, t: int) -> List[int]:
        return np.where(self.bad_mask[:, t])[0].tolist()

    def nodes_by_cause(self) -> Dict[str, List[NodeFinding]]:
        out: Dict[str, List[NodeFinding]] = {}
        for f in self.findings:
            out.setdefault(f.cause, []).append(f)
        return out

    @property
    def n(self) -> int:
        return len(self.findings)


def tier_threshold_ms(tier: str) -> float:
    return BUSINESS_TIERS.get(tier, (1.0, ""))[0] * 1e3


def build_thresholds(tier: List[str], is_server: np.ndarray) -> np.ndarray:
    th = np.array([tier_threshold_ms(t) for t in tier])
    th[is_server] = np.minimum(th[is_server], 10.0)
    return th


# ---------------------------------------------------------------- 主流程

def analyze(tel: dict, min_run_s: float = 1200.0, sigma_mult: float = 6.0,
            abs_floor_ms: float = 3.0) -> AnalysisResult:
    meta = tel["meta"]
    dev = np.asarray(tel["dev_ms"], dtype=np.float64)
    delay = np.asarray(tel["delay_ms"], dtype=np.float64)
    jitter = np.asarray(tel["jitter_ms"], dtype=np.float64)
    loss = np.asarray(tel["loss_pct"], dtype=np.float64)
    reach = np.asarray(tel["reach"], dtype=np.float64)
    names = meta["node_names"]
    tier = meta["node_tier"]
    region = meta["node_region"]
    is_server = np.asarray(meta["node_is_server"], dtype=bool)
    upstream_of = [list(u) for u in meta["upstream_of"]]
    times_s = np.asarray(tel["times_s"], dtype=np.float64)
    dt = float(meta["base_dt"])
    N, T = dev.shape

    th = build_thresholds(tier, is_server)

    # ---- 闰秒计划窗口 (整段 smear: ±13h)
    leap_rel = meta.get("leap_rel_s")
    scheduled = np.zeros((N, T), dtype=bool)
    if leap_rel is not None:
        half = int(13 * 3600 / dt)
        lo = max(0, int(leap_rel / dt) - half)
        hi = min(T, int(leap_rel / dt) + half + 1)
        scheduled[:, lo:hi] = True

    # ---- 基线 (排除闰秒窗口的样本)
    ok_samples = ~scheduled[0] if scheduled.any() else np.ones(T, dtype=bool)
    if not ok_samples.any():
        ok_samples = np.ones(T, dtype=bool)
    import warnings as _w
    with _w.catch_warnings(), np.errstate(all="ignore"):
        _w.simplefilter("ignore", RuntimeWarning)
        baseline = np.nanmedian(dev, axis=1)
        sigma = 1.4826 * np.nanmedian(np.abs(dev - baseline[:, None]), axis=1)
    baseline = np.where(np.isfinite(baseline), baseline, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.3)
    sigma = np.maximum(sigma, 0.3)

    limit = np.maximum(
        np.broadcast_to(th[:, None], (N, T)),
        np.maximum(baseline[:, None] + sigma_mult * sigma[:, None],
                   np.full((N, T), abs_floor_ms)))
    bad = (np.abs(dev) > limit) & (~scheduled)
    # 开局宽限: 前 1 小时是采集/收敛热身, 不判故障 (真实监控同样如此)
    grace = max(1, int(3600 / dt))
    bad[:, :grace] = False
    # 最短连续窗口
    min_len = max(1, int(round(min_run_s / dt)))
    bad_clean = _episode_filter(bad, min_len)

    # ---- 网络异常掩码
    net_bad = _network_bad(delay, jitter, loss, reach, region, dev, dt)

    # ---- 图
    children: List[List[int]] = [[] for _ in range(N)]
    for i, ups in enumerate(upstream_of):
        for u in ups:
            children[u].append(i)
    up_sets = [set(u) for u in upstream_of]
    # 兄弟缓存
    sib_summary = _sibling_summary(dev, bad_clean, children, times_s)

    # ---- 逐节点 finding
    findings: List[NodeFinding] = []
    for i in range(N):
        win = episodes(bad_clean[i], min_len=1)
        if not win:
            findings.append(NodeFinding(
                idx=i, name=names[i], tier=tier[i], region=region[i], cause="ok",
                peak_ms=0.0, rms_ms=0.0, onset_s=None, clear_s=None,
                culprit=None, confidence=1.0, evidence=["偏差在阈值内"]))
            continue
        ab = np.abs(dev[i])
        peak = float(np.nanmax(ab[bad_clean[i]]))
        rms = float(np.sqrt(np.nanmean(dev[i][bad_clean[i]] ** 2)))
        cause, culprit, conf, ev, subtype, pick = _attribute(
            i, dev, bad_clean, net_bad, up_sets, children, sib_summary,
            limit, is_server, tier[i], names, times_s)
        # onset 取归因所依据的那段窗口 (而非可能更早的无关脉冲)
        onset = float(times_s[pick[0]]); clear = float(times_s[pick[1] - 1])
        findings.append(NodeFinding(
            idx=i, name=names[i], tier=tier[i], region=region[i], cause=cause,
            peak_ms=peak, rms_ms=rms, onset_s=onset, clear_s=clear,
            culprit=culprit, confidence=conf, evidence=ev, subtype=subtype,
            windows=[(float(times_s[a]), float(times_s[b - 1])) for a, b in win]))

    up_incidents = _detect_upstream_incidents(
        dev, bad_clean, findings, children, names, times_s)
    net_incidents = _detect_network_incidents(net_bad, region, delay, jitter, loss, times_s)

    return AnalysisResult(
        threshold_ms=th, dev_ms=dev, bad_mask=bad_clean, findings=findings,
        upstream_incidents=up_incidents, network_incidents=net_incidents,
        baseline_ms=baseline, sigma_ms=sigma, node_names=names, tier=tier,
        region=region, times_s=times_s, meta=meta, scheduled_mask=scheduled)


def _episode_filter(bad: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(bad)
    for i in range(bad.shape[0]):
        for a, b in episodes(bad[i], min_len=min_len):
            out[i, a:b] = True
    return out


def _network_bad(delay, jitter, loss, reach, region, dev, dt) -> np.ndarray:
    N, T = dev.shape
    net = np.zeros((N, T), dtype=bool)
    # 区域基线
    d_base = np.zeros(N)
    j_base = np.zeros(N)
    l_base = np.zeros(N)
    for rg in set(region):
        ridx = np.array([j for j in range(N) if region[j] == rg])
        if ridx.size == 0:
            continue
        import warnings as _w
        with _w.catch_warnings(), np.errstate(all="ignore"):
            _w.simplefilter("ignore", RuntimeWarning)
            d_base[ridx] = np.nanmedian(delay[ridx], axis=1)
            j_base[ridx] = np.nanmedian(jitter[ridx], axis=1)
        l_base[ridx] = np.nanmedian(loss[ridx], axis=1)
    d_base = np.maximum(d_base, 0.5)
    j_base = np.maximum(j_base, 0.03)
    with np.errstate(all="ignore"):
        d_ratio = delay / d_base[:, None]
        j_ratio = jitter / j_base[:, None]
    # 真正的路径故障: 时延或抖动显著抬升, 或持续高丢包。单纯 reach 下降不算
    # (多数表决剔除单源、holdover 都会让 reach 掉, 那是源的问题不是链路)。
    net |= (d_ratio > 2.5) & (delay > 3.0)
    net |= (j_ratio > 5.0) & (jitter > 0.5)
    net |= loss > np.maximum(8.0, l_base[:, None] + 8.0)
    return _episode_filter(net, max(1, int(600 / dt)))


def _sibling_summary(dev, bad, children, times_s):
    """缓存每节点的兄弟列表与其 '共因强度'。"""
    return {"children": children}


def _attribute(i, dev, bad, net_bad, up_sets, children, sib_cache,
               limit, is_server, tier, names, times_s):
    """返回 (cause, culprit, confidence, evidence, subtype, window)。"""
    ev: List[str] = []
    my_bad = bad[i]
    wins = episodes(my_bad, min_len=1)
    if not wins:
        return "ok", None, 1.0, ["无越限窗口"], "", (0, 1)
    ups = sorted(up_sets[i])
    first_win = wins[0]

    # ---- 证据 1: 网络
    net_overlap = (my_bad & net_bad[i]).sum() / max(1, my_bad.sum())
    if net_overlap > 0.6:
        ev.append(f"偏差窗口与路径异常 {net_overlap * 100:.0f}% 重叠")
        # 取与网络异常重叠最多的那段窗口
        pick = max(wins, key=lambda w: (net_bad[i, w[0]:w[1]]).sum())
        return "network", None, min(0.95, 0.5 + 0.45 * net_overlap), ev, "path", pick

    # ---- 证据 2a: 本机瞬变 (VM 暂停恢复 / 人工改时 / ntpd 重启)
    # 判据: 窗口起点单步阶跃 > 200ms, 且阶跃后朝零单调回收 (摊开/步进纠正),
    # 而不是像 ntpd 停摆那样一路发散。这与窗口多长无关 (VM 若禁 step 需数小时才追回)。
    for a, b in wins:
        pre = dev[i, max(0, a - 1)]
        jump = abs(dev[i, a] - pre)
        if jump < 200.0:
            continue
        sibs = []
        for j in ups:
            sibs.extend([k for k in children[j] if k != i])
        sibs = list(set(sibs))
        if sibs:
            frac_sib_bad = np.mean([bad[k, a:b].any() for k in sibs])
        else:
            frac_sib_bad = 0.0
        # 上游共因会让一大片兄弟同时偏; 本机事件则只有零星几台
        if frac_sib_bad > 0.25:
            continue
        post = dev[i, a:min(len(dev[i]), b + 4)]
        # 窗口内偏差绝对值是否 "单调朝零走" (阶跃后步进纠正或 500ppm 摊平都会如此),
        # 还是 "先平后陡降" (人工 date -s 长时间保持再被纠正)。
        if len(post) >= 4:
            frac_dec = float(np.mean(np.diff(np.abs(post)) <= 0))
        else:
            frac_dec = 1.0
        ev.append(f"窗口起点单步阶跃 {jump:.0f}ms, 同上游兄弟全正常 -> 本机自身跳变")
        # 阶跃型本机故障: VM 暂停恢复、人工 date -s、ntpd 重启均为此形态。
        # 仅凭遥测难 100% 区分, 统一记为 local_jump, 并在证据里给出判别线索。
        if frac_dec >= 0.7:
            ev.append(f"阶跃后逐步回收 (回收段 {b - a} 采样, 递减占比 {frac_dec*100:.0f}%), 更像 VM 暂停恢复后重新收敛")
        else:
            ev.append("阶跃后较长时间保持再陡降, 更像人工改时后人工纠正")
        return "local", None, 0.85, ev, "local_jump", (a, b)

    # ---- 证据 2b: 上游共因
    best = None
    for a, b in wins:
        seg = slice(a, b)
        my_med = float(np.nanmedian(dev[i, seg]))
        my_dir = np.sign(my_med)
        my_mag = abs(my_med)
        for j in ups:
            sibs = [k for k in children[j] if k != i]
            if not sibs:
                continue
            sib_bad = [k for k in sibs if bad[k, seg].any()]
            if not sib_bad:
                continue
            frac = len(sib_bad) / len(sibs)
            dirs = [np.sign(np.nanmedian(dev[k, seg])) for k in sib_bad]
            dir_agree = float(np.mean([d == my_dir for d in dirs]))
            mags = [abs(np.nanmedian(dev[k, seg])) for k in sib_bad]
            mag_ratio = my_mag / max(1e-9, float(np.median(mags)))
            mag_ok = 1.0 if 0.33 < mag_ratio < 3.0 else 0.5
            score = frac * (0.5 + 0.5 * dir_agree) * mag_ok
            if bad[j, seg].any():
                score *= 1.15
            if best is None or score > best[0]:
                best = (score, j, frac, dir_agree, mag_ratio, (a, b), len(sibs), len(sib_bad))

    if best is not None and best[0] > 0.45:
        score, j, frac, dir_agree, mag_ratio, seg, nsib, nbad = best
        ev.append(f"共享上游 {names[j]} 的 {nsib} 台兄弟中 {nbad} 台 ({frac*100:.0f}%) 同期越限")
        ev.append(f"方向一致率 {dir_agree*100:.0f}%, 幅度比 {mag_ratio:.2f}")
        if bad[j, seg[0]:seg[1]].any():
            ev.append(f"上游 {names[j]} 自身同期越限")
        return "upstream", int(j), min(0.98, 0.52 + 0.45 * score), ev, "feed", seg

    # ---- 证据 3: 本机 (逐步漂移 / 长时间未同步)
    if net_overlap < 0.3:
        ev.append("同上游兄弟正常 / 无区域路径异常")
        ev.append(f"仅本机偏移, 峰 {abs(float(np.nanmax(dev[i][my_bad]))):.0f}ms")
        return "local", None, 0.72, ev, "dead_ntpd", first_win
    ev.append("证据不足")
    return "unknown", None, 0.3, ev, "", first_win


def _detect_upstream_incidents(dev, bad, findings, children, names, times_s,
                               gap_s: float = 3 * 3600):
    """按 (culprit, 连续时间段) 拆分事件: 同一台上游在不同时间段的两次故障
    (如先被 ntp-c 带偏、后来又自己失联) 应各算一次事件。"""
    out: List[UpstreamIncident] = []
    by_idx = {f.idx: f for f in findings}
    T = bad.shape[1]
    N = len(names)
    if T > 1:
        gap = max(1, int(gap_s / max(1.0, times_s[1] - times_s[0])))
    else:
        gap = 1
    for j in range(N):
        kids = children[j]
        if not kids:
            continue
        followers = [k for k in kids if by_idx[k].cause == "upstream" and by_idx[k].culprit == j]
        self_bad = bad[j]
        if len(followers) < 2 and not (self_bad.any() and len(followers) >= 1):
            continue
        # 把 followers 的坏窗口 + 自身坏窗口收集为时间区间, 合并成 <gap 的簇
        intervals = []
        for k in followers:
            intervals.extend(episodes(bad[k], min_len=1))
        if self_bad.any():
            intervals.extend(episodes(self_bad, min_len=1))
        if not intervals:
            continue
        intervals.sort()
        clusters = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a - clusters[-1][1] <= gap:
                clusters[-1][1] = max(clusters[-1][1], b)
            else:
                clusters.append([a, b])
        for (ca, cb) in clusters:
            seg_f = [k for k in followers if bad[k, ca:cb].any()]
            seg_self = bool(self_bad[ca:cb].any())
            if len(seg_f) < 2 and not (seg_self and len(seg_f) >= 1):
                continue
            onset = float(times_s[ca])
            peaks = [by_idx[k].peak_ms for k in seg_f]
            if seg_self:
                peaks.append(abs(float(np.nanmax(dev[j, ca:cb]))))
            peak = max(peaks) if peaks else 0.0
            dir_vals = [np.nanmedian(dev[k, ca:cb][bad[k, ca:cb]]) for k in seg_f if bad[k, ca:cb].any()]
            if seg_self:
                dir_vals.append(np.nanmedian(dev[j, ca:cb][self_bad[ca:cb]]))
            direction = float(np.sign(np.median(dir_vals))) if dir_vals else 0.0
            protected = len(kids) - len(seg_f)
            ev = [f"上游 {names[j]} 直连下游 {len(kids)} 台, 本时段 {len(seg_f)} 台同向偏"]
            if seg_self:
                ev.append(f"{names[j]} 自身偏差峰 {abs(float(np.nanmax(dev[j, ca:cb]))):.0f}ms")
            if protected > 0:
                ev.append(f"{protected} 台因多数表决/冗余未被带偏")
            conf = min(0.99, 0.5 + 0.5 * (len(seg_f) / max(1, len(kids))))
            out.append(UpstreamIncident(
                culprit=j, name=names[j], onset_s=onset, detected_s=onset,
                peak_ms=peak, direction=direction, direct_children=len(kids),
                affected=sorted(seg_f), protected=protected, confidence=conf, evidence=ev))
    out.sort(key=lambda x: -x.peak_ms)
    return out


def _detect_network_incidents(net_bad, region, delay, jitter, loss, times_s):
    out: List[NetworkIncident] = []
    for rg in sorted(set(region)):
        ridx = np.array([i for i in range(len(region)) if region[i] == rg])
        if ridx.size < 3:
            continue
        # 区域基线 (整段的中位)
        d_b = float(np.nanmedian(delay[ridx]))
        j_b = max(0.03, float(np.nanmedian(jitter[ridx])))
        col = net_bad[ridx].sum(axis=0) >= max(3, len(ridx) // 5)
        for a, b in episodes(col, min_len=3):
            seg = slice(a, b)
            sub = net_bad[ridx, seg]
            affected = [int(ridx[k]) for k in range(len(ridx)) if sub[k].any()]
            d_m = float(np.nanmedian(delay[ridx, seg])) if not np.isnan(delay[ridx, seg]).all() else d_b
            j_m = float(np.nanmedian(jitter[ridx, seg])) if not np.isnan(jitter[ridx, seg]).all() else j_b
            l_a = float(np.nanmedian(loss[ridx, seg])) if not np.isnan(loss[ridx, seg]).all() else 0.0
            dr = d_m / max(d_b, 1e-6)
            jr = j_m / max(j_b, 1e-6)
            # 至少一项指标显著恶化才算链路事件 (排除 holdover/批量不可达造成的空事件)
            if dr < 1.5 and jr < 2.0 and l_a < 5.0:
                continue
            out.append(NetworkIncident(
                key=rg, onset_s=float(times_s[a]), end_s=float(times_s[b - 1]),
                affected=affected, delay_ratio=dr, jitter_ratio=jr, loss_add=l_a,
                confidence=min(0.95, 0.4 + 0.1 * len(affected) / len(ridx)),
                evidence=[f"{rg} 区域 {b-a} 个采样多数节点路径异常",
                          f"时延基线 {d_b:.1f}ms -> {d_m:.1f}ms (×{dr:.1f}), "
                          f"抖动基线 {j_b:.2f}ms -> {j_m:.2f}ms, 丢包 {l_a:.0f}%"]))
    return out


# ---------------------------------------------------------------- 摘要

def summary(ar: AnalysisResult) -> dict:
    causes: Dict[str, int] = {}
    for f in ar.findings:
        causes[f.cause] = causes.get(f.cause, 0) + 1
    return {
        "n_nodes": len(ar.findings),
        "causes": causes,
        "n_bad": sum(causes.get(k, 0) for k in ("local", "upstream", "network", "unknown")),
        "upstream_incidents": [{
            "culprit": u.culprit, "name": u.name,
            "onset": _iso(ar, u.onset_s), "affected": len(u.affected),
            "protected": u.protected, "peak_ms": round(u.peak_ms, 2),
            "direction": u.direction, "confidence": round(u.confidence, 2),
            "direct_children": u.direct_children,
        } for u in ar.upstream_incidents],
        "network_incidents": [{
            "region": n.key, "onset": _iso(ar, n.onset_s), "end": _iso(ar, n.end_s),
            "affected": len(n.affected), "delay_ratio": round(n.delay_ratio, 2),
            "jitter_ratio": round(n.jitter_ratio, 2), "loss_add": round(n.loss_add, 1),
        } for n in ar.network_incidents],
    }


def _iso(ar: AnalysisResult, rel_s: float) -> str:
    from ..util import parse_utc
    from datetime import timedelta
    return fmt_utc(parse_utc(ar.meta["start"]) + timedelta(seconds=float(rel_s)))
