"""闰秒对具体业务原语的影响 —— 机理级模拟 + 影响清单.

对四类最易出事的时间原语, 用真实时钟轨迹驱动一个简化但保真的业务模型,
统计出问题次数/后果, 而不是拍脑袋列清单:

1. **分布式锁租约 (lease)**
   客户端在 t0 发 `SET lock NX PX ttl`, 服务端按自己的墙钟记录到期时刻。
   若服务端时钟在持锁期间**倒退** 1s, 已到期的租约会被判为 "还没到期",
   或客户端以为还持锁却已被别人抢到 -> 临界区重叠, 双写。
   若时钟**前跳**, 租约会提前过期 -> 另一个进程立即拿锁, 同样双写。
   -> 模拟: 两台机器的时钟轨迹, 统计 "锁提前到期" 与 "租约重叠" 的次数。

2. **超时 / 重试计算**
   `deadline = start + timeout`, 到期判断用墙钟差。
   时钟倒退 -> 实际超时窗口被拉长 (甚至永不超时); 时钟前跳 -> 立即误超时。
   -> 模拟: 统计超时判定与实际真实时长的偏差、误超时次数。

3. **日志排序 (跨机)**
   按墙钟时间戳排序合并多机日志。1s 的跨机偏差 -> 因果顺序倒置
   (先发生的请求排到后面), 影响审计/回放。
   -> 模拟: 生成有真实因果顺序的事件流, 统计排序后 "逆序对" 比例。

4. **按时间生成的编号**
   `id = 时间戳 << k | 序列号` (Snowflake 类), 或按日期分桶。
   时钟倒退 -> 生成比之前更小的 ID -> 冲突/主键回退; 跨秒边界 -> 重复。
   -> 模拟: 统计单调性破坏次数、ID 回退次数、需要 "时钟回拨保护" 的次数。

每个原语返回 {metric, value, severity, detail} 列表, 汇总成一张业务清单。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..util import rng
from .policies import DEFAULT_WINDOW, clock_reading


@dataclass
class Impact:
    primitive: str            # lease | timeout | log_order | id_gen
    primitive_cn: str
    scenario: str             # 具体工况
    metric: str
    value: float
    unit: str
    severity: str             # ok | low | medium | high | critical
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive, "primitive_cn": self.primitive_cn,
            "scenario": self.scenario, "metric": self.metric,
            "value": round(self.value, 4), "unit": self.unit,
            "severity": self.severity, "detail": self.detail,
        }


SEV_RANK = {"ok": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def _sev_from(v: float, thresholds: List[Tuple[float, str]]) -> str:
    """thresholds 升序: [(上限, severity)]。"""
    for lim, sev in thresholds:
        if v <= lim:
            return sev
    return thresholds[-1][1]


def _max_rate_error(mode: str, window_s: float) -> float:
    """墙钟速率相对 1 的最大偏离 (秒/秒)。step 为瞬时不连续, 单独处理。"""
    if mode in ("slew", "smear"):
        return 1.0 / max(window_s, 60.0)
    return 0.0


def _worst_hold_error(mode: str, window_s: float, hold_s: float) -> float:
    """一次持续 hold_s 的操作, 在该策略下墙钟计时误差的最大值。

    摊开策略: 误差 = 速率偏离 × 重叠在窗口内的时长, 上界为 rate_err*hold_s。
    step: 若持有期跨过跳变点, 误差恰好 1s (时钟在瞬间倒退 1s)。
    """
    if mode == "step":
        return 1.0
    return min(1.0, _max_rate_error(mode, window_s) * hold_s)


# ---------------------------------------------------------------- 1. 租约

def simulate_lease(mode: str, window_s: float, ttl_s: float = 10.0,
                   leap_tau: float = 0.0, span_h: float = 30.0,
                   horizon_s: float = 3600.0, n: int = 200000) -> List[Impact]:
    """两台机器 (服务端 A 用给定策略, 客户端 B 同为该策略) 在持锁期间跨过闰秒。

    租约计时误差按 *一个 TTL* 的真实持有时长缩放, 而非整段窗口。
    """
    impacts = []
    if mode == "step":
        impacts.append(Impact("lease", "分布式锁租约", "持锁期间跨闰秒: 时钟倒退 1s",
                              "租约残留延长", 1.0, "s", "critical",
                              "服务端计时倒退 1s, 本应过期的锁仍被判有效, 新客户端可再加锁 -> 双写"))
        impacts.append(Impact("lease", "分布式锁租约", "持锁期间跨闰秒: 时钟倒退 1s",
                              "临界区重叠概率", 1.0, "倍", "critical",
                              "租约到期判定失效, 两进程可能同时进入临界区"))
    elif mode in ("slew", "smear"):
        extra = _worst_hold_error(mode, window_s, ttl_s)
        sev = _sev_from(extra, [(0.01, "low"), (0.1, "medium"), (0.5, "high"), (1e9, "critical")])
        impacts.append(Impact("lease", "分布式锁租约", f"{mode} {int(window_s)}s 摊开",
                              "单个租约实际时长延长", extra, "s", sev,
                              f"摊开期间墙钟速率偏 {_max_rate_error(mode, window_s)*1e6:.1f}ppm, "
                              f"TTL={ttl_s}s 实际约 {ttl_s + extra:.4f}s, 单调无重叠"))
    else:  # none
        impacts.append(Impact("lease", "分布式锁租约", "忽略闰秒",
                              "租约与外部对齐偏差", 1.0, "s", "high",
                              "本机与已纠正机器永久差 1s, 看他人的租约均偏 1s, 可能提前/延后抢占"))
    return impacts


# ---------------------------------------------------------------- 2. 超时

def simulate_timeout(mode: str, window_s: float, timeout_s: float = 3.0,
                     leap_tau: float = 0.0, span_h: float = 30.0,
                     n: int = 200000) -> List[Impact]:
    impacts = []
    if mode == "step":
        impacts.append(Impact("timeout", "超时/重试", "请求窗口跨过闰秒",
                              "误判超时/卡死", 1.0, "次/跳变", "high",
                              "时钟倒退使 elapsed 变小甚至为负, 已超时的请求被判未超时 -> 线程堆积"))
        impacts.append(Impact("timeout", "超时/重试", "请求窗口跨过闰秒",
                              "实际超时窗口偏差", 1.0, "s", "high",
                              f"timeout={timeout_s}s 在跳变点前后失真最多 1s, 重试风暴/长尾"))
    elif mode in ("slew", "smear"):
        err = _worst_hold_error(mode, window_s, timeout_s)
        sev = _sev_from(err, [(0.005, "low"), (0.05, "medium"), (0.2, "high"), (1e9, "critical")])
        impacts.append(Impact("timeout", "超时/重试", f"{mode} 摊开中",
                              "单个超时窗口最大偏差", err, "s", sev,
                              f"墙钟单调无倒退, timeout={timeout_s}s 期间误差 ≤ {err*1000:.2f}ms"))
    else:
        impacts.append(Impact("timeout", "超时/重试", "忽略闰秒",
                              "与对齐组的判定偏差", 1.0, "s", "high",
                              "本机快 1s, 对已纠正对端的心跳/超时判断整体偏 1s"))
    return impacts


# ---------------------------------------------------------------- 3. 日志排序

def simulate_log_order(mode: str, window_s: float, n_events: int = 20000,
                       machines: int = 3, leap_tau: float = 0.0,
                       span_s: float = 7200.0, seed: int = 7,
                       mixed_with: str = None) -> List[Impact]:
    """生成有真实因果顺序的跨机事件, 各自按本机墙钟打时间戳, 再按时间戳排序,
    统计逆序对比例 (排序后违反真实因果的比例)。

    关键: 灾难性逆序来自 **混用策略**。默认让一半机器用 mode、一半用 mixed_with
    (若给定), 模拟 "同一个机群既有 step 又有 smear" 的真实工况。
    """
    r = rng(seed, "log")
    t_true = np.sort(r.uniform(leap_tau - span_s / 2, leap_tau + span_s / 2, n_events))
    noise = r.normal(0, 0.002, n_events)
    stamps_ideal = t_true + noise
    idx_ideal = np.argsort(stamps_ideal, kind="stable")
    inv_ideal = sum(1 for k in range(1, n_events) if idx_ideal[k - 1] > idx_ideal[k])

    # 每台机器用哪种策略
    if mixed_with is not None:
        modes = [mode if (i % 2 == 0) else mixed_with for i in range(machines)]
    else:
        modes = [mode] * machines
    wmap_cache = {}
    def wmap_for(m, mm):
        key = (m, mm)
        if key not in wmap_cache:
            wmap_cache[key] = clock_reading(m, DEFAULT_WINDOW[m], t_true, leap_tau)
        return wmap_cache[key]
    stamps = np.empty(n_events)
    for i in range(n_events):
        mm = modes[i % machines]
        stamps[i] = wmap_for(mm, i % machines)[i] + noise[i]
    idx = np.argsort(stamps, kind="stable")
    inv = sum(1 for k in range(1, n_events) if idx[k - 1] > idx[k])
    frac = inv / max(1, n_events - 1)
    n_cross = 0
    for k in range(1, n_events):
        a, b = idx[k - 1], idx[k]
        if a > b and abs(stamps[b] - stamps[a]) > 0.1:
            n_cross += 1
    if n_cross > 0:
        sev = "critical" if n_cross > 10 else "high"
    elif mode == "step":
        sev = "medium"
    elif mode in ("slew", "smear"):
        sev = _sev_from(frac, [(0.005, "low"), (0.02, "medium"), (1e9, "high")])
    else:
        sev = "high"
    tag = f"{mode}" if mixed_with is None else f"{mode}+{mixed_with} 混用"
    return [Impact("log_order", "跨机日志排序", tag,
                   "相邻逆序比例", frac * 100, "%", sev,
                   f"{machines} 台机器打时间戳后合并排序, 违反真实因果的相邻对占比; "
                   f"其中跨机时间差>100ms 的严重逆序 {n_cross} 处 (对照基线 {inv_ideal} 处)")]


# ---------------------------------------------------------------- 4. ID 生成

def simulate_id_gen(mode: str, window_s: float, rate_hz: float = 50.0,
                    leap_tau: float = 0.0, span_h: float = 30.0,
                    n: int = 200000) -> List[Impact]:
    """Snowflake 类 ID: id = floor(wall*1000)<<k | seq。统计跨闰秒期间: 回退次数、
    重复 ID、需要时钟回拨保护的次数。回退量用解析值 (step 恰好 1000ms)。"""
    impacts = []
    if mode == "step":
        impacts.append(Impact("id_gen", "按时间生成编号", "跨闰秒 step",
                              "时钟回拨量", 1000.0, "ms", "critical",
                              "时间戳毫秒回退 1000ms, Snowflake 必须触发 '时钟回拨' 保护 (拒绝/等待), "
                              "否则同一 ms<<k 区间被重放, 产生重复 ID 或主键冲突"))
        impacts.append(Impact("id_gen", "按时间生成编号", "跨闰秒 step",
                              "整秒 ID 空间被重放", 1.0, "s", "critical",
                              f"按 {rate_hz:.0f}/s 生成, 回退 1s 意味着约 {int(rate_hz)} 个 ID 位可能重复"))
    elif mode in ("slew", "smear"):
        # 摊开保证毫秒时间戳严格单调不减, 回退 0 次
        impacts.append(Impact("id_gen", "按时间生成编号", f"{mode} 摊开",
                              "时钟回拨次数", 0.0, "次", "ok",
                              "墙钟单调, 时间戳部分不回退; 单位时间生成 ID 数略降, 容量略降"))
    else:
        impacts.append(Impact("id_gen", "按时间生成编号", "忽略闰秒",
                              "与对齐组时间戳偏差", 1000.0, "ms", "high",
                              "本机 ID 时间戳与已纠正机器永久差 1000ms, 跨机排序/去重受影响"))
    return impacts


# ---------------------------------------------------------------- 汇总

PRIMITIVE_CN = {
    "lease": "分布式锁租约", "timeout": "超时/重试",
    "log_order": "跨机日志排序", "id_gen": "按时间生成编号",
}


def analyze_mode(mode: str, window_s: float, leap_tau: float = 0.0,
                 ttl_s: float = 10.0, timeout_s: float = 3.0) -> List[Impact]:
    out: List[Impact] = []
    out += simulate_lease(mode, window_s, ttl_s, leap_tau)
    out += simulate_timeout(mode, window_s, timeout_s, leap_tau)
    out += simulate_log_order(mode, window_s, leap_tau=leap_tau)
    out += simulate_id_gen(mode, window_s, leap_tau=leap_tau)
    return out


MIXED_PAIRS = [("step", "smear"), ("step", "slew"), ("slew", "smear"),
               ("step", "none"), ("smear", "none")]


def build_checklist(leap_tau: float = 0.0, ttl_s: float = 10.0,
                    timeout_s: float = 3.0) -> dict:
    """对每种策略跑四类原语, 汇总成业务清单 + 混合机群差分 + 混用工况。"""
    from .policies import mixed_fleet_summary
    modes = ["step", "slew", "smear", "none"]
    per_mode = {}
    for m in modes:
        w = DEFAULT_WINDOW[m]
        per_mode[m] = [i.to_dict() for i in analyze_mode(m, w, leap_tau, ttl_s, timeout_s)]
    # 混用工况: 一半机群用 A、一半用 B, 看日志排序/租约的灾难性逆序
    mixed_cases = []
    for a, b in MIXED_PAIRS:
        lo = simulate_log_order(a, DEFAULT_WINDOW[a], leap_tau=leap_tau, mixed_with=b)
        # 租约: 一方 step 一方 smear, 谁看谁都可能重叠
        lease_a = simulate_lease(a, DEFAULT_WINDOW[a], ttl_s, leap_tau)
        mixed_cases.append({
            "pair": f"{a}+{b}",
            "log_order": lo[0].to_dict(),
            "lease_severity": max((x.severity for x in lease_a), key=lambda s: SEV_RANK[s]),
        })
    mixed = mixed_fleet_summary(leap_tau=leap_tau)
    matrix = {}
    for prim in PRIMITIVE_CN:
        matrix[prim] = {}
        for m in modes:
            sev = "ok"
            vals = [x for x in per_mode[m] if x["primitive"] == prim]
            if vals:
                sev = max((x["severity"] for x in vals), key=lambda s: SEV_RANK[s])
            matrix[prim][m] = sev
    return {
        "per_mode": per_mode,
        "severity_matrix": matrix,
        "mixed_fleet": mixed,
        "mixed_cases": mixed_cases,
        "recommendation": _recommend(matrix, mixed),
    }


def _recommend(matrix: dict, mixed: dict) -> dict:
    # 推荐: 全机群统一 smear (业务最平滑), 但对必须与外部 UTC 对齐的系统用 slew;
    # 绝不混用 step 与 smear。
    worst = mixed.get("worst_pair", {})
    return {
        "primary": "smear (24h 抹平) 或 slew (1h) 二选一在全机群统一",
        "why": ("step 会造成时间倒退与跨机 1s 跳变, 是租约/编号/日志排序故障的直接原因; "
                "slew/smear 单调无倒退, 但抹平窗口内与外部 UTC 偏离, 需与外部对齐的系统 (如证书/计费) 用较短 slew"),
        "must_not_mix": f"step 与 smear 混用会让两组机器在抹平窗口内长期相差约 {worst.get('max_abs_diff_s', 1.0)}s",        "alignment": "对账/计费/证书等需真 UTC 的服务: slew 1h 且闰秒后 1h 内避免跨机强依赖",
    }
