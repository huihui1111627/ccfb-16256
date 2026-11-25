"""场景定义: 机群拓扑、时钟/网络参数、故障注入、闰秒事件.

设计原则
--------
仿真器产出两部分互不混用的数据:
  * **遥测 (observable)** —— 相当于生产里能采到的 `ntpq`/监控指标,
    分析模块只能看这些。
  * **真值 (truth)**     —— 被注入的故障、每台机器的真实时钟误差,
    只给评测用, 分析模块绝不读取。

节点统一编号: 0..S-1 为时间服务器 (先 stratum-1 后 stratum-2), S..S+M-1 为业务机器。
这样 "服务器" 既是别人的上游, 也可以有自己的上游, 图结构统一。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..util import UTC, rng

# ---------------------------------------------------------------- 业务分级

#: 业务对时间的敏感程度 -> (偏差告警阈值秒, 说明)
BUSINESS_TIERS = {
    "critical": (0.005, "交易/订单/分布式锁: 亚 10ms 级, 依赖租约与严格顺序"),
    "high": (0.050, "支付对账/风控: 50ms 级, 依赖超时与时间窗"),
    "standard": (0.500, "常规在线服务: 500ms 级, 日志与 TTL 为主"),
    "batch": (2.000, "离线批处理: 2s 级, 只关心小时级顺序"),
}

TIER_ORDER = ["critical", "high", "standard", "batch"]


# ---------------------------------------------------------------- 时钟模型

@dataclass
class OscillatorProfile:
    """晶振/时钟源特性。ppm 为相对标称频率的偏差范围。"""
    name: str
    ppm_lo: float        # 固有频偏下界 (ppm)
    ppm_hi: float        # 固有频偏上界
    wander: float        # 频偏随机游走强度 (ppm / sqrt(秒))
    p_bad: float = 0.0   # 落入 "坏晶振" 的比例 (频偏再放大 2.5~5 倍)
    bad_mult: Tuple[float, float] = (2.5, 5.0)


OSCILLATORS = {
    # 真实晶振: TCXO ±2ppm, 普通 XO ±30~50ppm, 廉价 ±100ppm, 虚拟化时钟 ±50~120ppm。
    # 坏晶振最多放大到 ~500ppm 以内 —— 超过 500ppm 时 ntpd 的 slew 上限也追不回。
    "tcxo":   OscillatorProfile("tcxo",   -2,   2,   1e-3, 0.001),
    "xo":     OscillatorProfile("xo",     -35,  35,  4e-3, 0.008),
    "cheap":  OscillatorProfile("cheap", -100, 100, 1.2e-2, 0.02),
    "vm":     OscillatorProfile("vm",    -60,  60,  8e-3, 0.01),
}


@dataclass
class LeapPolicy:
    """闰秒处置方式。"""
    mode: str            # step | slew | smear | none
    window_s: float = 0.0  # 摊开时长; step 时忽略

    def label(self) -> str:
        if self.mode == "step":
            return "直接跳变"
        if self.mode in ("slew", "smear"):
            return f"摊开{int(self.window_s)}s"
        return "忽略闰秒"


# ---------------------------------------------------------------- 拓扑

@dataclass
class Link:
    """i 向上游 j 取时。参数刻画该网络路径。"""
    src: int
    dst: int
    delay_ms: float = 1.0       # 单程时延
    asym_ms: float = 0.05       # 去回不对称 -> 测得的偏移偏差 ≈ asym/2
    jitter_ms: float = 0.05     # 每次测量的随机抖动 (1σ)
    loss: float = 0.0           # 丢包率
    region_path: str = ""       # 该链路经过的区域骨干 (用于网络故障注入)


@dataclass
class ServerNode:
    idx: int
    name: str
    stratum: int                # 1 = 外部上游, 2 = 内网中继
    region: str
    osc: OscillatorProfile
    leap: LeapPolicy
    upstream: List[int] = field(default_factory=list)
    ref_quality_s: float = 0.0005   # stratum-1 对参考源的跟踪精度 (秒)


@dataclass
class MachineNode:
    idx: int
    name: str
    region: str
    rack: str
    tier: str                   # 业务分级
    osc: OscillatorProfile
    leap: LeapPolicy
    upstream: List[int] = field(default_factory=list)
    is_vm: bool = False
    slew_only: bool = False      # ntpd 禁用了 step, 只能 500ppm 慢慢追


@dataclass
class Fault:
    """一次故障注入。kind 决定语义, 见 sim/clocksim.py。"""
    kind: str
    start: datetime
    end: datetime
    target: str = ""            # 节点名 / 区域 / "all"
    params: Dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def dur_s(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass
class LeapEvent:
    when: datetime
    # 每类节点的默认策略, 具体节点可在 topology 里覆盖
    default_policy: LeapPolicy = field(default_factory=lambda: LeapPolicy("smear", 86400))


@dataclass
class Scenario:
    name: str
    seed: int
    start: datetime
    days: float
    base_dt: float = 600.0          # 遥测采样间隔 (秒)
    n_machines: int = 2400
    servers: List[ServerNode] = field(default_factory=list)
    machines: List[MachineNode] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    faults: List[Fault] = field(default_factory=list)
    leap: Optional[LeapEvent] = None
    regions: List[str] = field(default_factory=list)
    n_regions_relay: int = 16

    # ---------------- 便捷属性
    @property
    def n_samples(self) -> int:
        return int(round(self.days * 86400 / self.base_dt))

    @property
    def times(self) -> List[datetime]:
        return [self.start + timedelta(seconds=i * self.base_dt) for i in range(self.n_samples)]

    @property
    def n_servers(self) -> int:
        return len(self.servers)

    @property
    def n_nodes(self) -> int:
        return len(self.servers) + len(self.machines)

    def node_name(self, idx: int) -> str:
        if idx < len(self.servers):
            return self.servers[idx].name
        return self.machines[idx - len(self.servers)].name

    def node_tier(self, idx: int) -> str:
        if idx < len(self.servers):
            return "server"
        return self.machines[idx - len(self.servers)].tier

    def node_region(self, idx: int) -> str:
        if idx < len(self.servers):
            return self.servers[idx].region
        return self.machines[idx - len(self.servers)].region

    def node_leap(self, idx: int) -> LeapPolicy:
        if idx < len(self.servers):
            return self.servers[idx].leap
        return self.machines[idx - len(self.servers)].leap

    def links_by_src(self) -> Dict[int, List[Link]]:
        out: Dict[int, List[Link]] = {}
        for ln in self.links:
            out.setdefault(ln.src, []).append(ln)
        return out

    # ---------------- 序列化
    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.strftime("%Y-%m-%dT%H:%M:%SZ")
        for f in d["faults"]:
            f["start"] = f["start"].strftime("%Y-%m-%dT%H:%M:%SZ")
            f["end"] = f["end"].strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.leap:
            d["leap"]["when"] = self.leap.when.strftime("%Y-%m-%dT%H:%M:%SZ")
        # asdict 已递归展开 dataclass; 只需把 osc 折回名字
        for s in d["servers"]:
            s["osc"] = s["osc"]["name"]
        for m in d["machines"]:
            m["osc"] = m["osc"]["name"]
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 生成

REGION_NAMES = ["cn-north-1a", "cn-north-1b", "cn-east-2a", "cn-east-2b",
                "cn-south-1a", "cn-south-1b", "ap-sg-1a", "eu-fra-1a"]


def _mk_links(sc: Scenario, r: np.random.Generator, links: List[Link],
              src: int, dsts: List[int], base: Tuple[float, float],
              region_path: str) -> None:
    for d in dsts:
        dly = float(r.uniform(*base))
        links.append(Link(
            src=src, dst=d,
            delay_ms=dly,
            asym_ms=float(abs(r.normal(0, 0.15)) + 0.02),
            jitter_ms=float(abs(r.normal(0.04, 0.02)) + 0.01),
            loss=float(abs(r.normal(0.0, 0.002))),
            region_path=region_path,
        ))


def build_scenario(
    seed: int = 20260105,
    n_machines: int = 2400,
    days: float = 21.0,
    base_dt: float = 600.0,
    start: Optional[datetime] = None,
    n_tier1: int = 6,
    n_relay_per_region: int = 2,
    leap_at_day: float = 14.5,
    inject: str = "full",
) -> Scenario:
    """构造一个完整场景。inject: none|basic|full 控制故障注入丰富度。"""
    r = rng(seed, "topology")
    start = start or datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC)
    regions = REGION_NAMES[:8]
    sc = Scenario(name=f"fleet-{n_machines}-{days:g}d", seed=seed, start=start,
                  days=days, base_dt=base_dt, n_machines=n_machines, regions=regions,
                  n_regions_relay=n_relay_per_region * len(regions))

    # ---- stratum-1 外部上游
    t1_specs = [
        ("ntp-a.pool", "cn-north-1a", "tcxo", LeapPolicy("smear", 86400)),
        ("ntp-b.pool", "cn-north-1a", "tcxo", LeapPolicy("smear", 86400)),
        ("ntp-c.pool", "cn-east-2a", "tcxo", LeapPolicy("step")),
        ("ntp-d.pool", "cn-south-1a", "tcxo", LeapPolicy("smear", 86400)),
        ("ntp-e.pool", "ap-sg-1a", "xo", LeapPolicy("slew", 3600)),
        ("ntp-f.pool", "eu-fra-1a", "xo", LeapPolicy("step")),
    ][:n_tier1]
    for i, (nm, reg, osc, lp) in enumerate(t1_specs):
        sc.servers.append(ServerNode(
            idx=i, name=nm, stratum=1, region=reg, osc=OSCILLATORS[osc],
            leap=lp, upstream=[], ref_quality_s=float(abs(r.normal(300e-6, 80e-6))),
        ))

    # ---- stratum-2 内网中继, 每区 2 台
    relays: List[ServerNode] = []
    for reg in regions:
        for k in range(n_relay_per_region):
            idx = len(sc.servers)
            # 中继尽量跟同区 tier1, 否则跟最近的
            same = [s.idx for s in sc.servers if s.region == reg]
            pool = same if same else list(range(n_tier1))
            ups = sorted(r.choice(pool, size=min(2, len(pool)), replace=False).tolist())
            # 再补一台异地 tier1 增加多样性
            others = [x for x in range(n_tier1) if x not in ups]
            if others:
                ups.append(int(r.choice(others)))
            lp = LeapPolicy("smear", 86400) if r.random() < 0.7 else LeapPolicy("step")
            relays.append(ServerNode(
                idx=idx, name=f"relay-{reg}-{k+1}", stratum=2, region=reg,
                osc=OSCILLATORS["tcxo" if k == 0 else "xo"],
                leap=lp, upstream=ups,
            ))
            sc.servers.append(relays[-1])
            for u in ups:
                _mk_links(sc, r, sc.links, idx, [u], (0.3, 3.0), f"{reg}->core")
    S = len(sc.servers)

    # ---- 业务机器
    # 刻意混入一段 "历史遗留区": 该区的部分机器只有单一路径, 且该路径最终只依赖
    # ntp-c 一台外部源 —— 一旦 ntp-c 出错, 这批机器就是共因故障的受害者。
    legacy_region = "cn-east-2a"
    legacy_relay_idx = len(sc.servers)
    sc.servers.append(ServerNode(
        idx=legacy_relay_idx, name="relay-legacy-1", stratum=2,
        region=legacy_region, osc=OSCILLATORS["xo"],
        leap=LeapPolicy("step"), upstream=[2],   # 单源, 无冗余 —— 错误配置
    ))
    _mk_links(sc, r, sc.links, legacy_relay_idx, [2], (0.5, 4.0), "cn-east-2a->core")
    S = len(sc.servers)

    leaf_osc = ["xo", "xo", "cheap", "vm"]
    for m in range(n_machines):
        reg = regions[int(r.integers(0, len(regions)))]
        rack = f"{reg}-r{int(r.integers(0, 12)):02d}"
        tier = r.choice(TIER_ORDER, p=[0.18, 0.27, 0.45, 0.10])
        osc = OSCILLATORS[leaf_osc[int(r.choice(len(leaf_osc), p=[0.45, 0.25, 0.15, 0.15]))]]
        is_vm = osc.name == "vm"
        if r.random() < 0.55:
            lp = LeapPolicy("smear", 86400)
        elif r.random() < 0.6:
            lp = LeapPolicy("step")
        else:
            lp = LeapPolicy("slew", float(r.choice([3600, 7200])))
        slew_only = bool(r.random() < 0.06)
        idx = S + m
        node = MachineNode(idx=idx, name=f"{reg}-{rack}-{m:05d}", region=reg, rack=rack,
                           tier=str(tier), osc=osc, leap=lp, is_vm=is_vm,
                           slew_only=slew_only)

        roll = r.random()
        if roll < 0.09:
            # 遗留配置: 只挂那一台单源中继 -> 无冗余, 共因受害者
            node.region = legacy_region
            node.upstream = [legacy_relay_idx]
        elif roll < 0.12:
            # 直连唯一一台外部源 ntp-c -> 同样无冗余
            node.upstream = [2]
        elif roll < 0.20:
            # 只配一台普通中继 (脆弱)
            local = [s.idx for s in relays if s.region == reg]
            node.upstream = [int(r.choice(local))] if local else [2]
        else:
            # 正常: 本区两台中继 + 30% 再直连一台 tier1
            local = [s.idx for s in relays if s.region == reg]
            ups = list(r.choice(local, size=min(2, len(local)), replace=False).tolist())
            if r.random() < 0.30:
                ups.append(int(r.integers(0, n_tier1)))
            node.upstream = sorted(set(int(u) for u in ups))
        sc.machines.append(node)
        for u in node.upstream:
            _mk_links(sc, r, sc.links, idx, [u], (0.15, 12.0), f"{node.region}")
        _ = tier

    # ---- 故障注入
    if inject != "none":
        sc.faults.extend(_make_faults(sc, r, inject, leap_at_day))
    if leap_at_day is not None and leap_at_day > 0:
        sc.leap = LeapEvent(when=start + timedelta(days=leap_at_day))
    return sc


def _make_faults(sc: Scenario, r: np.random.Generator, mode: str,
                 leap_at_day: float) -> List[Fault]:
    st = sc.start
    faults: List[Fault] = []
    S = len(sc.servers)
    # 1) 一台上游时间服务器自己走错: +240s 突跳 (对应故事里 "快了四分钟")
    bad_up = "ntp-c.pool"
    faults.append(Fault(
        kind="upstream_bad", target=bad_up,
        start=st + timedelta(days=7, hours=3, minutes=12),
        end=st + timedelta(days=9, hours=48) if False else st + timedelta(days=9, hours=0),
        params={"step_s": 240.0, "drift_ppm": 0.0},
        note="上游 GPS 失锁后时钟被手动改正 +4min, 之后未校正",
    ))
    # 2) 一批虚拟机从暂停恢复 -> 时间落后数十秒
    vm_idx = [m.idx for m in sc.machines if m.is_vm]
    n_sus = min(72, len(vm_idx))
    sus = sorted(np.random.default_rng(sc.seed + 7).choice(vm_idx, size=n_sus, replace=False).tolist())
    faults.append(Fault(
        kind="vm_suspend", target=",".join(str(i) for i in sus),
        start=st + timedelta(days=5, hours=10, minutes=0),
        end=st + timedelta(days=5, hours=10, minutes=0),  # 瞬时事件, dur 由 params 决定
        params={"suspend_s": 42.0, "stagger_s": 3.0},
        note="宿主机维护导致 72 台 VM 暂停 ~42s 后恢复, 客户机时钟未跟上",
    ))
    if mode == "basic":
        return faults
    # 3) 某区域骨干链路拥塞 -> 时延/抖动/丢包升高
    faults.append(Fault(
        kind="network_congest", target="cn-east-2a",
        start=st + timedelta(days=11, hours=2),
        end=st + timedelta(days=12, hours=6),
        params={"delay_mult": 4.0, "jitter_mult": 25.0, "loss_add": 0.25,
                "asym_add_ms": 3.0},
        note="cn-east-2a 出口拥塞: 时延×4, 抖动×25, 丢包 25%",
    ))
    # 4) 少数机器本机问题: ntpd 停摆自由漂移
    dead = [m.idx for m in sc.machines if m.tier in ("critical", "high")][:6]
    faults.append(Fault(
        kind="local_dead_ntpd", target=",".join(str(i) for i in dead),
        start=st + timedelta(days=3, hours=6),
        end=st + timedelta(days=6, hours=0),
        params={}, note="ntpd 进程被 OOM 杀掉且无守护拉起, 本机自由漂移",
    ))
    # 5) 一台机器被人工改错时间 (误操作)
    manual = [m.idx for m in sc.machines if m.tier == "standard"][:1]
    faults.append(Fault(
        kind="local_manual_set", target=str(manual[0]),
        start=st + timedelta(days=13, hours=8),
        end=st + timedelta(days=13, hours=20),
        params={"jump_s": -95.0}, note="运维排查问题时手动 date -s 改错 95s",
    ))
    # 6) 一台内网中继的上游全挂 -> 降级为自由运行 (stratum 16 兜底)
    relay = sc.servers[S - 1]
    faults.append(Fault(
        kind="relay_orphan", target=relay.name,
        start=st + timedelta(days=15, hours=1),
        end=st + timedelta(days=16, hours=0),
        params={}, note="该中继与其全部上游失联, 进入 holdover 自由漂移",
    ))
    return faults


DEFAULT = dict(seed=20260105, n_machines=2400, days=21.0, base_dt=600.0,
               n_tier1=6, n_relay_per_region=2, leap_at_day=14.5, inject="full")


def default_scenario(**kw) -> Scenario:
    args = dict(DEFAULT)
    args.update(kw)
    return build_scenario(**args)
