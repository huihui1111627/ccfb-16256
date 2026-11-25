"""通用小工具: 确定性随机数、稳健统计、时间格式化、区间抽取。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import numpy as np

UTC = timezone.utc


def rng(seed: int, salt: str = "") -> np.random.Generator:
    """由 (seed, salt) 派生确定性随机数发生器, 保证同一场景可复现。"""
    h = 0
    for ch in salt:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return np.random.default_rng((int(seed) * 1_000_003 + h) & 0x7FFFFFFF)


def robust_sigma(x: np.ndarray) -> float:
    """MAD 估计的标准差, 抗离群值。"""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """沿最后一维的滑动中位数 (末尾对齐), 边界用可用窗口。"""
    x = np.asarray(x)
    if win <= 1:
        return x.copy()
    n = x.shape[-1]
    out = np.empty_like(x, dtype=np.float64)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[..., i] = np.median(x[..., lo:hi], axis=-1)
    return out


def episodes(mask: Sequence[bool], min_len: int = 1) -> List[Tuple[int, int]]:
    """把布尔序列切成 [start, end) 连续区间。"""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def pct(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关, 自动忽略 NaN; 样本不足返回 0。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return 0.0
    a, b = a[ok], b[ok]
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 6) -> Tuple[float, int]:
    """在 ±max_lag 个采样内找最大滞后相关, 返回 (corr, lag)。lag>0 表示 b 领先。"""
    best = (0.0, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            aa, bb = a[lag:], b[:-lag]
        elif lag < 0:
            aa, bb = a[:lag], b[-lag:]
        else:
            aa, bb = a, b
        c = pearson(aa, bb)
        if abs(c) > abs(best[0]):
            best = (c, lag)
    return best


def parse_utc(s: str) -> datetime:
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def fmt_utc(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_dur(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 90:
        return f"{seconds:.1f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def fmt_offset(seconds: float) -> str:
    """带符号的偏移量, 自动选单位。"""
    sign = "+" if seconds >= 0 else "-"
    a = abs(float(seconds))
    if a < 1e-3:
        return f"{sign}{a * 1e6:.0f}µs"
    if a < 1:
        return f"{sign}{a * 1e3:.1f}ms"
    return f"{sign}{a:.3f}s"


class UnionFind:
    """并查集, 用于把同源受影响的机器聚成一簇。"""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self) -> dict:
        out: dict = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return out
