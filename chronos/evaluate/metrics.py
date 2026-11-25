"""评测: 把分析结论与仿真真值逐项对比。

指标
----
1. **故障机器识别**: 真值里 "被某故障影响" 的机器集合 vs 分析判为异常 (非 ok) 的集合
   -> precision / recall / F1。
2. **根因归因**: 对真值受影响机器, 按真值期望原因 (local / upstream / network) 与
   分析给出的 cause 做混淆矩阵, 统计每类准确率。
3. **问题原因 (机器自身 / 上游 / 网络)**: 总体三分类准确率。
4. **上游故障定位**: 出错的服务器是否被认出、detection latency (首个越限样本时刻),
   受影响的下游集合与真值的交集/并集。
5. **误报**: 真值无故障却在分析里被判异常的机器数。

真值 "受影响机器" 的判定: 在对应故障的时间窗内, |真实偏差| 超过其业务阈值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..analysis.attribution import AnalysisResult, tier_threshold_ms
from ..sim.scenario import BUSINESS_TIERS


def _rel(t: str) -> float:
    from ..util import parse_utc
    return 0.0 if not t else 0.0


@dataclass
class EvalResult:
    fault_node_eval: dict          # precision/recall/f1 for "is this node faulted"
    cause_confusion: Dict[Tuple[str, str], int]
    cause_accuracy: float
    per_class: Dict[str, dict]
    upstream_eval: List[dict]
    false_positive_nodes: List[int]
    false_negative_nodes: List[int]
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fault_detection": self.fault_node_eval,
            "cause_accuracy": round(self.cause_accuracy, 4),
            "cause_confusion": {f"{a}->{b}": v for (a, b), v in self.cause_confusion.items() if v},
            "per_class": self.per_class,
            "upstream": self.upstream_eval,
            "n_false_positive": len(self.false_positive_nodes),
            "n_false_negative": len(self.false_negative_nodes),
        }


class Epoch:
    """真值 epoch 访问器。"""

    def __init__(self, truth: dict, meta: dict):
        self.truth = truth
        self.meta = meta
        self.true_off = np.asarray(truth["true_off_s"], dtype=np.float64)   # [N,T] 秒
        self.times_s = np.asarray(meta["_times_s"], dtype=np.float64) if "_times_s" in meta else None
        self.fault_meta = truth["fault_meta"]
        self.base_dt = float(meta["base_dt"])
        self.n_servers = int(meta["n_servers"])
        self.tier = meta["node_tier"]
        self.names = meta["node_names"]
        self.upstream_of = [list(u) for u in meta["upstream_of"]]
        # 每个故障的精确影响掩码 (仿真器直接产出, 比启发式更可信)
        self.masks: Dict[str, np.ndarray] = {
            k[len("mask__"):]: v for k, v in truth.items() if k.startswith("mask__")
        }

    def th_ms(self, i: int) -> float:
        t = self.tier[i]
        return BUSINESS_TIERS.get(t, (1.0, ""))[0] * 1e3

    def timeline(self) -> np.ndarray:
        return np.arange(self.true_off.shape[1]) * self.base_dt

    def window_idx(self, start_s: float, end_s: float) -> Tuple[int, int]:
        T = self.true_off.shape[1]
        a = max(0, int(round(start_s / self.base_dt)))
        b = min(T, max(a + 1, int(round(end_s / self.base_dt))))
        return a, b

    def affected_by_fault(self, f: dict) -> Set[int]:
        """由故障掩码 + "因果下游 + 实际越限" 精确确定受影响节点。

        掩码本身只标了直接被注入的节点 (如坏掉的那台上游服务器), 不会自动包含
        被它带偏的下游。所以这里把掩码节点沿依赖关系向下传播, 再要求该节点在
        故障窗内确实越限 —— 二者同时满足才算 "真值受影响"。
        """
        a, b = self.window_idx(f["start_s"], f["end_s"])
        key = self._mask_key(f)
        seg = np.abs(self.true_off[:, a:b])
        N = self.true_off.shape[0]
        # 本机类故障 (vm 暂停/人工改时/停摆): 掩码覆盖整个故障生命期, 用全时间轴
        use_full_mask = f["kind"] not in ("upstream_bad", "relay_orphan", "network_congest")
        if key is not None and key in self.masks:
            mk = self.masks[key]
            roots = np.where((mk if use_full_mask else mk[:, a:b]).any(axis=1))[0]
        else:
            roots = np.arange(N)
        downstream = self._downstream(set(int(r) for r in roots)) if f["kind"] in (
            "upstream_bad", "relay_orphan") else set(int(r) for r in roots)
        # 故障自身导致的偏差量级; 一个 +240s 的阶跃不可能让节点只偏 0.5s,
        # 因此要求受影响节点的偏差达到该量级的一定比例, 以排除恰好同期的本机漂移。
        mag_floor = self._fault_magnitude(f)
        out = set()
        for i in downstream:
            need = max(self.th_ms(i) / 1000.0 * 1.5, mag_floor)
            if use_full_mask:
                # 本机类: 在掩码命中的任一时刻越限即可 (窗口边界不可靠)
                mask_i = mk[i]
                hit = (np.abs(self.true_off[i]) > need) & mask_i
                if hit.any():
                    out.add(i)
            elif (seg[i] > need).any():
                out.add(i)
        return out

    def _fault_magnitude(self, f: dict) -> float:
        """故障的典型偏差量级 (秒); 无明确量级返回 0。"""
        p = f.get("params", {}) or {}
        if f["kind"] == "upstream_bad":
            return abs(float(p.get("step_s", 0.0))) * 0.10
        if f["kind"] == "local_manual_set":
            return abs(float(p.get("jump_s", 0.0))) * 0.10
        if f["kind"] == "vm_suspend":
            return abs(float(p.get("suspend_s", 0.0))) * 0.10
        return 0.0

    def _downstream(self, roots: Set[int]) -> Set[int]:
        """所有 (传递地) 依赖任一 root 的节点。"""
        if not hasattr(self, "_children"):
            ch: Dict[int, List[int]] = {}
            for i, ups in enumerate(self.upstream_of):
                for u in ups:
                    ch.setdefault(u, []).append(i)
            self._children = ch
        children = self._children
        seen = set(roots)
        stack = list(roots)
        while stack:
            x = stack.pop()
            for c in children.get(x, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def _mask_key(self, f: dict) -> Optional[str]:
        # 掩码 key 形如 "<kind>_<index>"
        idx = f.get("_idx")
        if idx is None:
            return None
        return f"{f['kind']}_{idx}"

    def exact_cause(self, i: int) -> Optional[str]:
        """节点 i 的真实主因: 取 "因果覆盖且窗内确实越限" 的故障中幅度最大者。"""
        best = None
        sev = 0.0
        for f in self.fault_meta:
            key = self._mask_key(f)
            if key is None or key not in self.masks:
                continue
            a, b = self.window_idx(f["start_s"], f["end_s"])
            if b <= a:
                continue
            roots = np.where(self.masks[key][:, a:b].any(axis=1))[0]
            if f["kind"] in ("upstream_bad", "relay_orphan"):
                if i not in self._downstream(set(int(r) for r in roots)):
                    continue
            elif i not in set(int(r) for r in roots):
                continue
            peak = float(np.nanmax(np.abs(self.true_off[i, a:b])))
            need = max(self.th_ms(i) / 1000.0 * 1.5, self._fault_magnitude(f))
            if peak < need:
                continue
            cause = _kind_to_cause(f["kind"])
            if cause is None:
                continue
            if peak > sev:
                sev = peak
                best = cause
        return best

    def all_faulted(self) -> Set[int]:
        s = set()
        for f in self.fault_meta:
            s |= self.affected_by_fault(f)
        return s


def _reachable(src: int, dst: int, upstream_of: List[List[int]]) -> bool:
    """dst 是否沿上游链依赖 src。"""
    seen = set()
    stack = [dst]
    while stack:
        x = stack.pop()
        if x == src:
            return True
        if x in seen:
            continue
        seen.add(x)
        for u in upstream_of[x]:
            stack.append(u)
    return False


def _kind_to_cause(kind: str) -> Optional[str]:
    return {
        "upstream_bad": "upstream",
        "relay_orphan": "upstream",
        "vm_suspend": "local",
        "local_dead_ntpd": "local",
        "local_manual_set": "local",
        "network_congest": "network",
    }.get(kind)


# ---------------------------------------------------------------- 主评测

def evaluate(ar: AnalysisResult, truth: dict) -> EvalResult:
    meta = ar.meta
    ep = Epoch(truth, {**meta, "_times_s": ar.times_s.tolist()})
    n = len(ar.findings)
    pred_bad = {f.idx for f in ar.findings if f.cause in ("local", "upstream", "network", "unknown")}
    true_bad = ep.all_faulted()

    tp = len(pred_bad & true_bad)
    fp = pred_bad - true_bad
    fn = true_bad - pred_bad
    prec = tp / max(1, tp + len(fp))
    rec = tp / max(1, tp + len(fn))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    fault_eval = {
        "n_true": len(true_bad), "n_pred": len(pred_bad),
        "tp": tp, "fp": len(fp), "fn": len(fn),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }

    # 归因: 只对真值受影响机器
    confusion: Dict[Tuple[str, str], int] = {}
    correct = 0
    per_class_stat: Dict[str, dict] = {}
    for i in true_bad:
        exp = ep.exact_cause(i) or "unknown"
        got = ar.findings[i].cause
        confusion[(exp, got)] = confusion.get((exp, got), 0) + 1
        st = per_class_stat.setdefault(exp, {"n": 0, "correct": 0})
        st["n"] += 1
        if got == exp:
            st["correct"] += 1
            correct += 1
    for k, st in per_class_stat.items():
        st["accuracy"] = round(st["correct"] / max(1, st["n"]), 4)
    cause_acc = correct / max(1, len(true_bad))

    # 上游故障定位
    up_eval = []
    for inc in ar.upstream_incidents:
        # 找真值里对应的上游故障
        gt = None
        for f in ep.fault_meta:
            if f["kind"] in ("upstream_bad", "relay_orphan") and f["target"] == inc.name:
                gt = f
                break
        if gt is None:
            up_eval.append({"name": inc.name, "matched_ground_truth": False,
                            "affected": len(inc.affected)})
            continue
        # 真值: 用与 affected_by_fault 相同的因果 + 量级口径
        gt_affected = ep.affected_by_fault(gt) - {inc.culprit}
        pred_set = set(inc.affected)
        tp2 = len(pred_set & gt_affected)
        prec2 = tp2 / max(1, len(pred_set))
        rec2 = tp2 / max(1, len(gt_affected))
        latency = inc.detection_latency_s
        up_eval.append({
            "name": inc.name, "matched_ground_truth": True,
            "onset_iso_s": round(inc.onset_s, 1),
            "gt_start_iso_s": round(gt["start_s"], 1),
            "latency_s": round(latency, 1),
            "onset_error_s": round(inc.onset_s - gt["start_s"], 1),
            "affected_pred": len(pred_set), "affected_gt": len(gt_affected),
            "affected_precision": round(prec2, 4), "affected_recall": round(rec2, 4),
            "detected_as_culprit": True,
        })
    # 真值上游故障是否都至少被认出一个 culprit
    gt_up = [f for f in ep.fault_meta if f["kind"] in ("upstream_bad", "relay_orphan")]
    detected_names = {u["name"] for u in up_eval if u.get("matched_ground_truth")}
    missed = [f["target"] for f in gt_up if f["target"] not in detected_names]
    for f in gt_up:
        pass
    up_eval.append({"missed_ground_truth": missed}) if missed else None

    # 网络事件匹配
    net_match = []
    gt_net = [f for f in ep.fault_meta if f["kind"] == "network_congest"]
    for f in gt_net:
        hit = None
        for inc in ar.network_incidents:
            ov = max(0.0, min(inc.end_s, f["end_s"]) - max(inc.onset_s, f["start_s"]))
            dur = max(1.0, f["end_s"] - f["start_s"])
            if ov / dur > 0.3:
                hit = inc
                break
        net_match.append({
            "region": f["target"],
            "detected": hit is not None,
            "delay_ratio": round(hit.delay_ratio, 2) if hit else None,
            "jitter_ratio": round(hit.jitter_ratio, 2) if hit else None,
            "onset_error_s": round(hit.onset_s - f["start_s"], 1) if hit else None,
        })

    return EvalResult(
        fault_node_eval=fault_eval, cause_confusion=confusion,
        cause_accuracy=cause_acc, per_class=per_class_stat,
        upstream_eval=up_eval, false_positive_nodes=sorted(fp),
        false_negative_nodes=sorted(fn),
        detail={"network": net_match,
                "true_bad_by_cause": _count_true(true_bad, ep)},
    )


def _count_true(nodes: Set[int], ep: Epoch) -> dict:
    out = {}
    for i in nodes:
        c = ep.exact_cause(i) or "unknown"
        out[c] = out.get(c, 0) + 1
    return out
