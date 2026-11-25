"""闰秒策略的时钟轨迹 (语义严格版).

情景: **正闰秒** (insert) —— 真实世界插入一秒 (23:59:60)。
以真实流逝时间 τ (单调, 秒; 类似 TAI) 为横轴, w(τ) 为机器墙钟读数。

正闰秒后, 正确的 UTC 标签比 τ 永久落后 1 秒。各处置方式对 w 的影响:

* ``step`` : 经典 Linux/ntpd SET —— 闰秒瞬间把时钟**向后拨 1 秒**,
             w 出现 −1s 的阶跃 (τ≥0 时 w = τ − 1)。**时间倒流**, 单调性被破坏。
             这正是 "闰秒时服务出问题" 的典型来源。
* ``slew`` : 内核在 W 秒内线性把 1 秒摊掉 (w 速率 1 − 1/W), w 单调、连续,
             没有任何倒退, 峰值偏差 ≤1s。W 常见 1000~3600s。
* ``smear``: 同 slew, 但 W = 86400s (24h 抹平)。速率扰动仅 ~11.6ppm,
             但对需与外部 UTC 对齐的系统会持续偏离近 24h。
* ``none`` : 完全不处理, w = τ 恒定 -> 相对已纠正的机器永久**快 1 秒**。

关键推论:
  - 只有 step 会产生 **时间倒退** (单调性破坏);
  - slew / smear 不倒退, 但会在一段时间内与 step 机器相差最多 1 秒;
  - 全机群若 step 与 smear 混用, 在 smear 窗口内两组相差可长期维持 ~1s;
  - none 与任何已纠正组永久差 1s。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

POLICY_MODES = ["step", "slew", "smear", "none"]

MODE_LABEL = {
    "step": "直接跳变 step (向后拨 1s, 时间倒流)",
    "slew": "内核摊开 slew (线性, 不倒退)",
    "smear": "全域抹平 smear (24h, 不倒退)",
    "none": "忽略闰秒 none (永久快 1s)",
}

DEFAULT_WINDOW = {"step": 0.0, "slew": 3600.0, "smear": 86400.0, "none": 0.0}


def clock_reading(mode: str, window_s: float, tau: np.ndarray,
                  leap_tau: float = 0.0) -> np.ndarray:
    """墙钟读数 w(τ), 单位秒。leap_tau 为闰秒在真实时间轴上的时刻。"""
    tau = np.asarray(tau, dtype=np.float64)
    d = tau - leap_tau
    if mode == "step":
        return tau + np.where(d >= 0, -1.0, 0.0)
    if mode in ("slew", "smear"):
        W = max(window_s, 60.0)
        # d ∈ [0, W]: 偏差从 0 线性走到 -1 (w 略慢于 τ), 之后恒定 -1
        dev = -np.clip(d / W, 0.0, 1.0)
        return tau + dev
    if mode == "none":
        return tau.copy()
    raise ValueError(mode)


def deviation(mode: str, window_s: float, tau: np.ndarray, leap_tau: float = 0.0) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return clock_reading(mode, window_s, tau, leap_tau) - tau


@dataclass
class CurveSet:
    tau: np.ndarray
    readings: Dict[str, np.ndarray]
    devs: Dict[str, np.ndarray]
    labels: Dict[str, str]


def build_curves(window_s: float = 86400, span_h: float = 52.0, n: int = 8000,
                 leap_tau: float = 0.0) -> CurveSet:
    tau = np.linspace(-span_h / 2 * 3600, span_h / 2 * 3600, n)
    readings, devs, labels = {}, {}, {}
    for mode in POLICY_MODES:
        w = clock_reading(mode, window_s, tau, leap_tau)
        readings[mode] = w
        devs[mode] = w - tau
        labels[mode] = MODE_LABEL[mode]
    return CurveSet(tau=tau, readings=readings, devs=devs, labels=labels)


# ---------------------------------------------------------------- 单机指标

def curve_metrics(mode: str, window_s: float, leap_tau: float = 0.0,
                  span_h: float = 52.0, n: int = 200000) -> dict:
    """单台机器在该策略下的时钟健康度。阶跃用解析值, 避免采样踩点误差。"""
    tau = np.linspace(-span_h / 2 * 3600, span_h / 2 * 3600, n)
    w = clock_reading(mode, window_s, tau, leap_tau)
    dw = np.diff(w)
    dt = float(tau[1] - tau[0])
    rate = dw / dt
    backward = dw < -1e-6
    max_back_step = float(-np.min(dw)) if np.min(dw) < 0 else 0.0
    if mode == "step":
        max_back_step = 1.0                    # 解析: 恰好倒退 1s
    return {
        "mode": mode, "window_s": window_s,
        "max_abs_dev_s": round(float(np.max(np.abs(w - tau))), 4),
        "final_dev_s": round(float(w[-1] - tau[-1]), 4),
        "min_rate": round(float(np.min(rate)), 6),
        "max_rate": round(float(np.max(rate)), 6),
        "monotonic": bool(not backward.any() and max_back_step == 0.0),
        "max_backward_step_s": round(max_back_step, 4),
        "max_rate_error_ppm": round(float(np.max(np.abs(rate - 1.0))) * 1e6, 2),
    }


# ---------------------------------------------------------------- 混合机群差分

def fleet_differential(combos: List[Tuple[str, str]], window_s: float = None,
                       leap_tau: float = 0.0, span_h: float = 52.0,
                       n: int = 400000) -> List[dict]:
    """任意两种策略并存的机群, 两台机器墙钟读数之差 |w_a − w_b| 的峰值与持续时长。

    时间轴取 [闰秒 − 1h, 闰秒 + span_h], 确保覆盖抹平窗口内的中段差分,
    而不是只看两端 (两端各策略都已收敛颉 1s, 会掩盖中段差异)。
    每种策略用其默认窗口 (slew 1h / smear 24h), 除非 window_s 显式指定覆盖两者。
    """
    tau = np.linspace(leap_tau - 3600, leap_tau + span_h * 3600, n)
    dt = float(tau[1] - tau[0])
    out = []
    cache = {m: clock_reading(m, DEFAULT_WINDOW[m] if window_s is None else window_s,
                              tau, leap_tau) for m in POLICY_MODES}
    for a, b in combos:
        diff = np.abs(cache[a] - cache[b])
        out.append({
            "a": a, "b": b,
            "max_abs_diff_s": round(float(diff.max()), 4),
            "dur_gt_100ms_s": round(float((diff > 0.1).sum() * dt), 1),
            "dur_gt_500ms_s": round(float((diff > 0.5).sum() * dt), 1),
            "dur_gt_1s_s": round(float((diff > 1.0 - 1e-9).sum() * dt), 1),
        })
    return out


def mixed_fleet_summary(window_s: float = None, leap_tau: float = 0.0) -> dict:
    combos = [("step", "slew"), ("step", "smear"), ("slew", "smear"),
              ("step", "none"), ("slew", "none"), ("smear", "none")]
    rows = fleet_differential(combos, window_s, leap_tau)
    worst = max(rows, key=lambda r: (r["max_abs_diff_s"], r["dur_gt_500ms_s"]))
    return {"pairs": rows, "worst_pair": worst}
