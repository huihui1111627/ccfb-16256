"""时钟仿真器.

物理与协议模型
--------------
* **晶振**: 每台机器有固有频偏 `freq` (ppm) 与随机游走 `freq += wander·√dt·N(0,1)`。
* **本机时钟**: `off` = 本机时钟相对 "真参考时刻" 的偏差(秒)。
      每步  off += (freq - flt)·dt
* **NTP 伺服 (PI 环)**: 每步对所选上游做一次测量
      m = off_i - off_j + asym/2 + jitter·N(0,1)
  然后   off_i -= k·m ;  flt_i += β·m/dt  (并限幅 ±500ppm)
  当 |m| 超过 step 阈值 (默认 128ms) 且本机允许跳变时, 直接一步到位 (k=1);
  禁用 step 的机器 (`-x`) 则受 500ppm 最大 slew 限制 —— 240s 的误差要 5.5 天才能追回。
* **stratum-1**: 无上游, 视为锁定参考源, off 由 OU 过程决定, 稳定度 = ref_quality_s。
* **holdover**: 上游全不可达时停止伺服, 自由漂移。
* **网络**: 每条链路带单程时延、去回不对称、抖动、丢包; 拥塞故障会同时抬升四者。
* **VM 暂停**: 恢复瞬间本机时钟整体落后, 之后由伺服慢慢追(能否追上取决于是否允许跳变)。
* **闰秒**: 每台按自己的策略 (跳变/摊开/忽略) 产生相对 UTC 的偏差, 最大 1s。

只产出 **可观测遥测**; 被注入的故障与真实偏差另存为真值 (truth), 供评测模块使用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..util import rng
from .scenario import LeapPolicy, Scenario

STEP_THR_S = 0.128          # ntpd 默认 step 阈值
MAX_SLEW = 500e-6           # 最大 slew 速率 500ppm
K_PHASE = 0.90              # 相位环增益
BETA_FREQ = 0.30            # 频偏环增益
MONITOR_SIGMA_S = 0.3e-3    # 巡检参考的测量噪声 (σ)
MEDIAN_TOL_S = 0.010        # 中位数表决的离群门限: 与中位数相差 > 10ms 视为 falseticker


@dataclass
class SimResult:
    """仿真输出。telemetry 面向分析, truth 面向评测。"""
    scenario: Scenario
    times_s: np.ndarray                 # [T] 相对 start 的秒
    node_names: List[str]
    node_is_server: np.ndarray          # [N] bool
    node_stratum: np.ndarray            # [N]
    node_tier: List[str]
    node_region: List[str]

    # ---- 可观测遥测 ----
    dev_ms: np.ndarray                  # [N,T] 相对巡检参考的偏差 (ms)
    sync_off_ms: np.ndarray             # [N,T] 本机自报对所选上游的 offset (ms)
    delay_ms: np.ndarray                # [N,T] 所选上游路径时延 (ms)
    jitter_ms: np.ndarray               # [N,T] 路径抖动 (ms)
    loss_pct: np.ndarray                # [N,T] 最近 1h 丢包率 (%)
    reach: np.ndarray                   # [N,T] 8 次探测的 reach 值 0..255
    peer_idx: np.ndarray                # [N,T] 所选上游节点下标, -1 = 无
    upstream_of: List[List[int]]        # 每节点的上游节点下标

    # ---- 真值 (分析模块禁用) ----
    true_off_s: np.ndarray              # [N,T] 真实偏差 (秒)
    leap_off_s: np.ndarray              # [N,T] 闰秒导致的偏差分量 (秒)
    fault_mask: Dict[str, np.ndarray]   # fault_key -> [N,T] bool 该故障在此刻影响该节点
    fault_meta: List[dict] = field(default_factory=list)

    # 便捷索引
    @property
    def T(self) -> int:
        return self.times_s.size

    @property
    def N(self) -> int:
        return self.dev_ms.shape[0]


# ------------------------------------------------------------------ 闰秒分量

def leap_profile(policy: LeapPolicy, t_rel: np.ndarray, leap_rel: float) -> np.ndarray:
    """节点相对 "真 UTC" 的闰秒偏差 (秒), 按策略不同。t_rel 为相对 start 的秒。"""
    out = np.zeros_like(t_rel, dtype=np.float64)
    if leap_rel is None or not np.isfinite(leap_rel):
        return out
    if policy.mode == "step":
        # 瞬间插入/删除, 采样粒度下几乎不可见
        return out
    if policy.mode in ("smear", "slew"):
        W = max(policy.window_s, 60.0)
        half = W / 2.0
        d = np.abs(t_rel - leap_rel)
        inside = d <= half
        # 三角: 中心处偏离最大 0.5s, 两端归零 (正闰秒 -> 时钟略慢于 UTC)
        out = np.where(inside, -0.5 * (1.0 - d / half), 0.0)
        return out
    if policy.mode == "none":
        # 完全忽略闰秒: 自闰秒起固定偏 1s, 直到 3h 后被外部纠正
        out = np.where((t_rel >= leap_rel) & (t_rel < leap_rel + 3 * 3600), 1.0, 0.0)
        return out
    return out


# ------------------------------------------------------------------ 主循环

def simulate(sc: Scenario, seed: Optional[int] = None, verbose: bool = False) -> SimResult:
    r = rng(seed if seed is not None else sc.seed, "clocksim")
    S = len(sc.servers)
    M = len(sc.machines)
    N = S + M
    T = sc.n_samples
    dt = sc.base_dt
    times_s = np.arange(T, dtype=np.float64) * dt

    # ---- 节点属性
    node_names = [n.name for n in sc.servers] + [n.name for n in sc.machines]
    is_server = np.zeros(N, dtype=bool)
    is_server[:S] = True
    is_stratum1 = np.zeros(N, dtype=bool)
    for s in sc.servers:
        if s.stratum == 1 or not s.upstream:
            is_stratum1[s.idx] = True
    node_stratum = np.zeros(N, dtype=np.int16)
    for s in sc.servers:
        node_stratum[s.idx] = s.stratum
    for m in sc.machines:
        node_stratum[m.idx] = 3
    node_tier = [sc.node_tier(i) for i in range(N)]
    node_region = [sc.node_region(i) for i in range(N)]

    # ---- 上游矩阵 (填充到 max_u)
    ups: List[List[int]] = []
    for i in range(N):
        ups.append(list(sc.servers[i].upstream) if i < S else list(sc.machines[i - S].upstream))
    max_u = max(1, max(len(u) for u in ups))

    up_idx = np.full((N, max_u), -1, dtype=np.int32)
    lk_delay = np.zeros((N, max_u))
    lk_asym = np.zeros((N, max_u))
    lk_jitter = np.zeros((N, max_u))
    lk_loss = np.zeros((N, max_u))
    link_of = {}
    for ln in sc.links:
        link_of[(ln.src, ln.dst)] = ln
    for i in range(N):
        for k, j in enumerate(ups[i][:max_u]):
            ln = link_of.get((i, j))
            up_idx[i, k] = j
            if ln is not None:
                lk_delay[i, k] = ln.delay_ms
                lk_asym[i, k] = ln.asym_ms
                lk_jitter[i, k] = ln.jitter_ms
                lk_loss[i, k] = ln.loss
            else:  # 无链路记录则用默认
                lk_delay[i, k] = 1.0
                lk_asym[i, k] = 0.05
                lk_jitter[i, k] = 0.05

    # ---- 晶振初值
    freq = np.zeros(N)
    flt = np.zeros(N)
    nom_freq = np.zeros(N)                      # 标称频偏, OU 以此为中心
    for i in range(N):
        osc = sc.servers[i].osc if i < S else sc.machines[i - S].osc
        f = float(r.uniform(osc.ppm_lo, osc.ppm_hi)) * 1e-6
        if r.random() < osc.p_bad:
            f *= float(r.uniform(*osc.bad_mult)) * (1 if f >= 0 else -1)
        freq[i] = f
        nom_freq[i] = f
    is_vm = np.zeros(N, dtype=bool)
    slew_only = np.zeros(N, dtype=bool)
    for m in sc.machines:
        is_vm[m.idx] = m.is_vm
        slew_only[m.idx] = m.slew_only
    wander = np.array([(sc.servers[i].osc.wander if i < S else sc.machines[i - S].osc.wander)
                       for i in range(N)]) * 1e-6
    ref_q = np.zeros(N)
    for s in sc.servers:
        if is_stratum1[s.idx]:
            ref_q[s.idx] = s.ref_quality_s

    # ---- 闰秒
    leap_rel = None
    if sc.leap is not None:
        leap_rel = (sc.leap.when - sc.start).total_seconds()
    leap_policies = [sc.node_leap(i) for i in range(N)]
    leap_off = np.zeros((N, T))
    if leap_rel is not None:
        for i in range(N):
            leap_off[i] = leap_profile(leap_policies[i], times_s, leap_rel)

    # ---- 故障预处理
    no_corr = np.zeros((N, T), dtype=bool)      # 本步禁止伺服 (ntpd 停摆/人工禁用)
    ref_target = np.zeros(N)                    # stratum-1 锁定参考源的真实偏移
    step_jump: Dict[int, List[tuple]] = {}      # 步号 -> [(node, 秒)]
    net_link_boost = np.ones((T, 4))            # 每步 (delay_mul, jitter_mul, loss_add, asym_add)
    net_link_boost[:, 2:] = 0.0
    # 链路按 "区域骨干" 归类, 拥塞按区域生效 (不再是全局)
    region_list = sorted(set(sc.regions) | set(node_region))
    reg_code = {rg: k for k, rg in enumerate(region_list)}
    R = len(region_list)
    link_rcode = np.full((N, max_u), -1, dtype=np.int32)
    for li, ln in enumerate(sc.links):
        # 找到该链路在矩阵中的位置
        if ln.src < N and ln.dst in ups[ln.src]:
            k = ups[ln.src].index(ln.dst)
            link_rcode[ln.src, k] = reg_code.get(ln.region_path, reg_code.get(node_region[ln.src], -1))
    cong_tab = np.ones((R, T, 4))
    cong_tab[:, :, 2:] = 0.0
    # 每节点对所选上游的可达性额外封锁
    unreach = np.zeros((N, T), dtype=bool)
    ref_sched: Dict[int, List[tuple]] = {}       # 步号 -> [(node, ref_target 变化量)]
    fault_mask: Dict[str, np.ndarray] = {}
    fault_meta: List[dict] = []

    def _tidx(t: float) -> int:
        return int(round(t / dt))

    for fi, f in enumerate(sc.faults):
        key = f"{f.kind}@{fi}"
        mask = np.zeros((N, T), dtype=bool)
        a = max(0, _tidx((f.start - sc.start).total_seconds()))
        b = min(T, _tidx((f.end - sc.start).total_seconds()))
        if f.kind in ("upstream_bad",):
            tn = [s.idx for s in sc.servers if s.name == f.target]
            jump = f.params.get("step_s", 240.0)
            for i in tn:
                # 参考源本身偏了 +jump: 该服务器锁定在错误参考上, 有效时钟随之偏移
                ref_sched.setdefault(a, []).append((i, jump))
                ref_sched.setdefault(min(T - 1, b), []).append((i, -jump))
                mask[i, a:b] = True
        elif f.kind == "vm_suspend":
            nodes = [int(x) for x in f.target.split(",") if x]
            sus = f.params.get("suspend_s", 42.0)
            stag = f.params.get("stagger_s", 3.0)
            ee = rng(sc.seed + 11, "suspend")
            for i in nodes:
                t0 = a + int(ee.integers(-3, 4))
                t0 = max(0, min(T - 1, t0))
                j = -abs(float(r.normal(sus, stag)))
                step_jump.setdefault(t0, []).append((i, j))
                # 恢复后客户机时钟未跟上, 且 ntpd 需重启/重新收敛: 观测到一个持续窗口
                hold = int(ee.integers(2, 6))
                hi = min(T, t0 + hold + 1)
                no_corr[i, t0:hi] = True
                mask[i, t0:hi] = True
        elif f.kind == "network_congest":
            reg = f.target
            rc = reg_code.get(reg, -1)
            for li, ln in enumerate(sc.links):
                if ln.region_path == reg:
                    mask[ln.src, a:b] = True
            if rc >= 0:
                cong_tab[rc, a:b, 0] = np.maximum(cong_tab[rc, a:b, 0], f.params.get("delay_mult", 1.0))
                cong_tab[rc, a:b, 1] = np.maximum(cong_tab[rc, a:b, 1], f.params.get("jitter_mult", 1.0))
                cong_tab[rc, a:b, 2] = np.maximum(cong_tab[rc, a:b, 2], f.params.get("loss_add", 0.0))
                cong_tab[rc, a:b, 3] = np.maximum(cong_tab[rc, a:b, 3], f.params.get("asym_add_ms", 0.0))
            # 标记区域内节点
            for i in range(N):
                if node_region[i] == reg:
                    mask[i, a:b] = True
        elif f.kind == "local_dead_ntpd":
            nodes = [int(x) for x in f.target.split(",") if x]
            for i in nodes:
                no_corr[i, a:b] = True
                mask[i, a:b] = True
        elif f.kind == "local_manual_set":
            nodes = [int(x) for x in f.target.split(",") if x]
            for i in nodes:
                step_jump.setdefault(a, []).append((i, f.params.get("jump_s", -95.0)))
                no_corr[i, a:b] = True
                mask[i, a:b] = True
        elif f.kind == "relay_orphan":
            i = next((s.idx for s in sc.servers if s.name == f.target), -1)
            if i >= 0:
                no_corr[i, a:b] = True
                unreach[i, a:b] = True
                mask[i, a:b] = True
                # 其下游跟着受影响
                for j in range(N):
                    if i in ups[j]:
                        mask[j, a:b] = True
        fault_mask[key] = mask
        fault_meta.append({"key": key, "kind": f.kind, "target": f.target,
                           "start_s": a * dt, "end_s": b * dt,
                           "note": f.note, "params": f.params, "_idx": fi})

    # ---- 状态
    # strat-1 用 OU 稳态初始化; 其余节点也先取一个 ≤ 数十 ms 的随机初偏,
    # 避免 "全部从 0 出发" 造成的开局瞬态被误认为故障。
    off = r.normal(0, 0.5e-3, N)
    off[is_stratum1] = r.normal(0, np.maximum(ref_q[is_stratum1], 1e-6))

    # 记录缓冲
    dev_ms = np.zeros((N, T), dtype=np.float32)
    sync_off_ms = np.zeros((N, T), dtype=np.float32)
    delay_arr = np.zeros((N, T), dtype=np.float32)
    jitter_arr = np.zeros((N, T), dtype=np.float32)
    ok_arr = np.zeros((N, T), dtype=bool)
    peer_arr = np.full((N, T), -1, dtype=np.int16)
    true_off = np.zeros((N, T), dtype=np.float32)

    step_thr = np.full(N, STEP_THR_S)
    ou_noise = np.zeros(N)                       # stratum-1 相对其参考源的 OU 抖动
    theta = 1.0 / 600.0                          # 10 分钟回拉时间常数
    theta_f = 1.0 / 86400.0                      # 频偏 1 天回拉 (温漂/老化)
    for t in range(T):
        # 0) 参考源变更 (上游服务器被改错 / 恢复)
        for (i, delta) in ref_sched.get(t, []):
            ref_target[i] += delta
        # 1) 注入瞬时跳变
        for (i, j) in step_jump.get(t, []):
            off[i] += j

        # 2) 推进本机时钟
        drift = (freq - flt) * dt
        off = off + drift
        # 频偏围绕标称值做 OU (温漂/老化), 不会无限游走
        freq = nom_freq + (freq - nom_freq) * (1.0 - theta_f * dt) \
            + r.normal(0, 1, N) * wander * np.sqrt(dt)
        # stratum-1 锁定参考 (有效时钟 = 参考源偏移 + OU 抖动, 稳态标准差 = ref_q)
        s1 = is_stratum1
        if s1.any():
            sd = np.sqrt(2.0 * theta) * ref_q[s1] * np.sqrt(dt)
            ou_noise[s1] = ou_noise[s1] * (1.0 - theta * dt) + r.normal(0, 1, s1.sum()) * sd
            off[s1] = ref_target[s1] + ou_noise[s1]

        # 3) 网络增强 (按链路所属区域查拥塞表)
        if link_rcode.size and (cong_tab[:, t, 0] > 1.0).any():
            rc = np.maximum(link_rcode, 0)
            dmult_l = cong_tab[rc, t, 0]
            jmult_l = cong_tab[rc, t, 1]
            ladd_l = cong_tab[rc, t, 2]
            aadd_l = cong_tab[rc, t, 3]
        else:
            dmult_l = np.ones_like(lk_delay)
            jmult_l = np.ones_like(lk_delay)
            ladd_l = np.zeros_like(lk_delay)
            aadd_l = np.zeros_like(lk_delay)
        # 4) 测量
        valid = up_idx >= 0
        has = np.zeros(N, dtype=bool)
        sel = np.zeros(N, dtype=np.int64)
        peer_sel_off = np.zeros(N)               # 采纳的合成 offset (秒)
        peer_choice = np.full(N, -1, dtype=np.int64)
        if valid.any():
            uidx = np.where(valid, up_idx, 0)
            asym = (lk_asym + aadd_l) * 1e-3         # ms -> s
            jit = (lk_jitter * jmult_l) * 1e-3
            m = (off[:, None] - off[uidx]) + asym * 0.5
            m = m + r.normal(0, 1, m.shape) * np.maximum(jit, 1e-6)
            loss = lk_loss + ladd_l
            got = (r.random(m.shape) >= loss) & valid
            # 当前步各节点是否 "可用" (自身能取时; stratum-1 始终可用)
            usable = got.any(axis=1) | is_stratum1
            if unreach[:, t].any():
                got[unreach[:, t], :] = False
                usable = usable & (~unreach[:, t])
                usable[is_stratum1 & (~unreach[:, t])] = True
            # 上游必须可用, 否则该链路本步作废 —— 由低层向高层一次收敛
            for si in (2, 3):
                grp = node_stratum == si
                if grp.any():
                    pu = usable[uidx]
                    usable[grp] = (got[grp] & pu[grp]).any(axis=1)
            got = got & usable[uidx]
            # 中位数多数表决 (falseticker 剔除): 只有落在中位数附近的多数的才参与合成
            mf = np.where(got, m, np.nan)
            import warnings as _w
            with _w.catch_warnings(), np.errstate(all="ignore"):
                _w.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(mf, axis=1)
            inl = got & (np.abs(m - np.nan_to_num(med)[:, None]) <= MEDIAN_TOL_S)
            ninl = inl.sum(axis=1)
            nval = got.sum(axis=1)
            # NTP 交叉算法: 只有当 被剔除者 < 有效数的一半 时才算收敛, 否则判定为 "无多数" holdover。
            outliers = nval - ninl
            ok_sync = (nval >= 1) & (outliers < (nval + 1) // 2) & np.isfinite(med)
            with _w.catch_warnings(), np.errstate(all="ignore"):
                _w.simplefilter("ignore", RuntimeWarning)
                combined = np.nanmedian(np.where(inl, m, np.nan), axis=1)
            # 选中 "最接近合成值" 的上游, 仅用于展示/诊断
            scol = np.where(got, np.abs(m - np.nan_to_num(combined)[:, None]), np.inf)
            sel = np.argmin(scol, axis=1)
            has = ok_sync & (~is_stratum1)
            peer_sel_off = np.where(ok_sync, combined, 0.0)
            peer_choice = np.where(got.any(axis=1), uidx[np.arange(N), sel], -1)
            # ---- 路径指标: 对所有节点 (含 stratum-1 无上游者) 取可用链路均值
            w = got.astype(np.float64)
            wsum = w.sum(axis=1)
            with np.errstate(all="ignore"):
                delay_mean = (w * lk_delay * dmult_l).sum(axis=1) / np.maximum(wsum, 1)
                jit_mean = (w * lk_jitter * jmult_l).sum(axis=1) / np.maximum(wsum, 1)
            has_link = wsum > 0
        else:
            m = np.zeros((N, 1))
            delay_mean = np.full(N, np.nan)
            jit_mean = np.full(N, np.nan)
            has_link = np.zeros(N, dtype=bool)
        # 5) 伺服
        if has.any():
            msel = peer_sel_off
            active = has & (~no_corr[:, t])
            # 大误差: 允许跳变则 step
            big = np.abs(msel) > step_thr
            can_step = active & big & (~slew_only)
            slew = active & big & slew_only
            small = active & (~big)
            # step: 直接对齐 (k=1), 不污染频偏估计
            off[can_step] -= msel[can_step]
            # slew-only 机器即便大幅偏差也只能以 500ppm 追赶 (真实 ntpd -x / VM 默认)
            off[slew] -= np.clip(msel[slew], -MAX_SLEW * dt, MAX_SLEW * dt)
            # 小误差: PI
            off[small] -= K_PHASE * msel[small]
            flt[small] += BETA_FREQ * msel[small] / dt
            flt = np.clip(flt, -MAX_SLEW, MAX_SLEW)
            # 所有活跃节点都按 PI 学习频偏 (slew-only 也学, 但慢)
            flt[slew] += np.clip(BETA_FREQ * msel[slew] / dt, -MAX_SLEW / 20, MAX_SLEW / 20)
            flt = np.clip(flt, -MAX_SLEW, MAX_SLEW)
        # 6) 记录
        monitor_noise = r.normal(0, MONITOR_SIGMA_S, N)
        dev_ms[:, t] = (off + leap_off[:, t] + monitor_noise) * 1e3
        true_off[:, t] = off + leap_off[:, t]
        if has.any():
            sync_off_ms[:, t] = np.where(has, peer_sel_off * 1e3, np.nan)
        delay_arr[:, t] = np.where(has_link, delay_mean, np.nan)
        jitter_arr[:, t] = np.where(has_link, jit_mean, np.nan)
        peer_arr[:, t] = peer_choice.astype(np.int16)
        # 可达性: 有上游的节点才能判定 "已同步"; stratum-1 视为始终可达
        ok_arr[:, t] = (has_link & (~is_stratum1)) | is_stratum1
        _ = t
        if verbose and t % 500 == 0:
            print(f"  sim {t}/{T} t={times_s[t] / 86400:.1f}d")

    # ---- 由原始 ok 计算 reach / loss
    # 采样间隔 10min, 真正的链路丢包率要在足够多次探测上统计才有意义;
    # 用 6h 窗口 (36 个采样) 估计丢包率, 避免 "单个探测失败=16.7%" 的假象。
    reach = np.zeros((N, T), dtype=np.float32)
    loss_pct = np.zeros((N, T), dtype=np.float32)
    win8 = 8
    win_loss = max(4, int(6 * 3600 / dt))
    for t in range(T):
        lo8 = max(0, t - win8 + 1)
        reach[:, t] = ok_arr[:, lo8:t + 1].mean(axis=1) * 255
        lo1 = max(0, t - win_loss + 1)
        seg = ok_arr[:, lo1:t + 1]
        loss_pct[:, t] = (1 - seg.mean(axis=1)) * 100

    return SimResult(
        scenario=sc, times_s=times_s, node_names=node_names,
        node_is_server=is_server, node_stratum=node_stratum,
        node_tier=node_tier, node_region=node_region,
        dev_ms=dev_ms, sync_off_ms=sync_off_ms, delay_ms=delay_arr,
        jitter_ms=jitter_arr, loss_pct=loss_pct, reach=reach,
        peer_idx=peer_arr, upstream_of=ups,
        true_off_s=true_off, leap_off_s=leap_off,
        fault_mask=fault_mask, fault_meta=fault_meta,
    )
