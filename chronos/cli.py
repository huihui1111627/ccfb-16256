"""chronos 命令行入口.

  python -m chronos gen            # 生成机群遥测数据集
  python -m chronos analyze        # 偏差检测 + 根因归因
  python -m chronos leap           # 闰秒推演 + 业务清单
  python -m chronos scan [DIR]     # 扫描业务代码墙上时钟用法
  python -m chronos advise         # 对时配置建议
  python -m chronos eval           # 与真值对比评测
  python -m chronos serve          # 启动网页
  python -m chronos report         # 一次性打印完整摘要
"""
from __future__ import annotations

import argparse
import os
import sys




def _c(txt, code):
    if not sys.stdout.isatty():
        return txt
    return f"\033[{code}m{txt}\033[0m"


def _bold(t): return _c(t, "1")
def _red(t): return _c(t, "31")
def _grn(t): return _c(t, "32")
def _yel(t): return _c(t, "33")
def _blu(t): return _c(t, "34")
def _dim(t): return _c(t, "2")


def cmd_gen(args):
    from .sim import build_and_save
    print(_bold(f"生成机群仿真数据 -> {args.out}/{args.tag}"))
    kwargs = dict(n_machines=args.machines, days=args.days, base_dt=args.dt,
                  inject="full" if not args.no_faults else "none")
    sc, res, paths = build_and_save(args.out, args.tag, **kwargs)
    print(f"  节点 {sc.n_nodes} (服务器 {sc.n_servers} + 业务机 {len(sc.machines)})")
    print(f"  采样 {res.T} 点 × {res.N} 节点, {args.days:g} 天")
    print(f"  故障注入 {len(sc.faults)} 处")
    if sc.leap:
        print(f"  闰秒 @ {sc.leap.when:%Y-%m-%d %H:%M} UTC")
    print(_grn("  已写入 ") + paths["telemetry"].replace(".npz", ".*"))


def _load(args):
    from .sim.dataset import load_telemetry
    prefix = os.path.join(args.out, args.tag)
    return load_telemetry(prefix + ".telemetry.npz")


def cmd_analyze(args):
    from .analysis import analyze, summary
    tel = _load(args)
    ar = analyze(tel)
    s = summary(ar)
    print(_bold("偏差巡检与根因归因"))
    print(f"  节点 {s['n_nodes']}, 异常 {_red(str(s['n_bad']))} 台")
    cc = s["causes"]
    print(f"    本机 {_yel(str(cc.get('local',0)))} · 上游 {_red(str(cc.get('upstream',0)))}"
          f" · 网络 {_blu(str(cc.get('network',0)))} · 正常 {_grn(str(cc.get('ok',0)))}")
    print()
    if s["upstream_incidents"]:
        print(_bold(_red("上游故障:")))
        for u in s["upstream_incidents"]:
            print(f"  ✗ {_bold(u['name'])}  起始 {u['onset']}  峰值 {u['peak_ms']:.0f}ms")
            print(f"      直连 {u['direct_children']} 台 · 实际带偏 {_red(str(u['affected']))} 台"
                  f" · 多数表决/冗余保护 {u['protected']} 台 · 置信度 {u['confidence']}")
    if s["network_incidents"]:
        print(_bold(_blu("网络故障:")))
        for n in s["network_incidents"]:
            print(f"  ⚠ {_bold(n['region'])}  {n['onset']} → {n['end']}  "
                  f"时延×{n['delay_ratio']} 抖动×{n['jitter_ratio']} 丢包+{n['loss_add']}%"
                  f"  影响 {n['affected']} 台")


def cmd_leap(args):
    from .leap import build_checklist
    cl = build_checklist()
    print(_bold("闰秒推演"))
    print("  严重度矩阵 (原语 × 策略):")
    sev = {"critical": _red("致命"), "high": _c("高", "31"), "medium": _yel("中"),
           "low": _blu("低"), "ok": _grn("无")}
    hdr = f"    {'原语':<14}{'step':<8}{'slew':<8}{'smear':<8}{'none':<8}"
    print(hdr)
    cn = {"lease": "锁租约", "timeout": "超时/重试", "log_order": "日志排序", "id_gen": "时间编号"}
    for p, row in cl["severity_matrix"].items():
        line = f"    {cn[p]:<14}"
        for m in ["step", "slew", "smear", "none"]:
            line += f"{sev[row[m]]:<8}"
        print(line)
    print()
    print(_bold("混用策略的灾难 (跨机事件排序):"))
    for c in cl["mixed_cases"]:
        import re
        m = re.search(r"严重逆序 (\d+)", c["log_order"]["detail"])
        n = m.group(1) if m else "?"
        print(f"    {c['pair']:<14} 逆序 {c['log_order']['value']:.2f}%  严重逆序 {n} 处  "
              f"租约 {sev[c['lease_severity']]}")
    print()
    print(_bold("推荐: ") + cl["recommendation"]["primary"])
    for f in cl["recommendation"]["must_not_mix"].split("，"):
        print("    " + _yel("⚠ " + f))


def cmd_scan(args):
    from .scan.wallclock import scan_dir, summarize
    from .evaluate.scanner_eval import evaluate_scan
    root = args.dir
    findings = scan_dir(root)
    s = summarize(findings)
    print(_bold(f"扫描 {root}: {s['total']} 处问题"))
    print("  按规则:", s["by_rule"])
    sev_cn = {"critical": _red, "high": _c, "medium": _yel, "low": _blu}
    for f in sorted(findings, key=lambda x: -{"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x.severity, 0))[:args.limit]:
        c = sev_cn.get(f.severity, lambda x: x)
        print(f"  {c(f.severity.upper()):<10} {f.path.split('/')[-1]}:{f.lineno}  {f.rule}")
        print(f"      {_dim(f.snippet)}")
        print(f"      → {f.suggestion}")
    if args.eval_corpus and os.path.isdir(args.eval_corpus):
        ev = evaluate_scan(args.eval_corpus)
        print()
        print(_bold("扫描器评测 (人工标注语料):"))
        print(f"  精确率 {ev['precision']*100:.1f}%  召回 {ev['recall']*100:.1f}%  "
              f"F1 {ev['f1']*100:.1f}%  漏报 {ev['fn']}  误报 {ev['fp']}")


def cmd_advise(args):
    from .advise import build_advice
    tel = _load(args)
    a = build_advice(tel["meta"])
    t = a["topology"]
    print(_bold("对时配置建议"))
    print(_bold("1. 上游编排"))
    print(f"   {t['summary']}")
    for ex in t["examples"][:args.limit]:
        col = _red if ex["severity"] == "critical" else _yel
        print(f"   {col('●')} {ex['name']}")
        print(f"     问题: {'; '.join(ex['issues'])}")
        print(f"     建议上游: {ex['current']} → {_grn(str(ex['recommended']))}")
    print()
    print(_bold("2. 闰秒策略"))
    print(f"   全机群统一: {_grn(a['leap']['uniform_choice'])}")
    for f in a["leap"]["forbidden"]:
        print("   " + _red("✕ " + f))
    print()
    print(_bold("3. 分级阈值"))
    for k, v in a["thresholds"]["tiers"].items():
        print(f"   {k:<10} {v['threshold_ms']:>6.0f} ms  窗口 {v['window_s']}s  {v['action']}")


def cmd_eval(args):
    from .sim.dataset import load_truth
    from .analysis import analyze
    from .evaluate import evaluate
    tel = _load(args)
    truth = load_truth(os.path.join(args.out, args.tag) + ".telemetry.npz")
    ar = analyze(tel)
    ev = evaluate(ar, truth).to_dict()
    fd = ev["fault_detection"]
    def _p(v):
        return "N/A" if v is None else f"{v * 100:.1f}%"
    print(_bold("与真值对比评测"))
    print(f"  故障机器识别: 精确率 {_p(fd['precision'])}  召回 {_p(fd['recall'])}  "
          f"F1 {_p(fd['f1'])}")
    print(f"    (真值 {fd['n_true']}, 判出 {fd['n_pred']}, 命中 {fd['tp']}, "
          f"误报 {fd['fp']}, 漏报 {fd['fn']})")
    acc = ev["cause_accuracy"]
    print("  根因归因准确率: " + (_grn(f"{acc * 100:.1f}%") if fd["n_true"] else "N/A"))
    for cls, st in ev["per_class"].items():
        print(f"    {cls:<10} {st['correct']}/{st['n']} = {st['accuracy']*100:.1f}%")
    print(_bold("  上游故障定位:"))
    for u in ev["upstream"]:
        if u.get("matched_ground_truth"):
            print(f"    ✓ {u['name']}  起始误差 {u['onset_error_s']:.0f}s  "
                  f"受影响 {u['affected_pred']}台 (精确 {u['affected_precision']*100:.0f}% "
                  f"召回 {u['affected_recall']*100:.0f}%)")
        elif u.get("explained_by"):
            print(f"    ~ {u['name']}  ({u['affected']}台) 是 {u['explained_by']} 的传播")
        elif u.get("missed_ground_truth"):
            print(f"    ✗ 漏检: {u['missed_ground_truth']}")


def cmd_serve(args):
    from .web.app import app
    print(_bold(f"启动 http://127.0.0.1:{args.port}"))
    app.run(host="127.0.0.1", port=args.port, debug=False)


def cmd_report(args):
    cmd_analyze(args)
    print()
    cmd_eval(args)
    print()
    cmd_leap(args)


def main(argv=None):
    p = argparse.ArgumentParser(prog="chronos", description="机群对时巡检与推演")
    p.add_argument("--out", default="data", help="数据目录")
    p.add_argument("--tag", default="fleet", help="数据集标签")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="生成仿真数据")
    g.add_argument("--machines", type=int, default=2400)
    g.add_argument("--days", type=float, default=21.0)
    g.add_argument("--dt", type=float, default=600.0)
    g.add_argument("--no-faults", action="store_true")
    g.set_defaults(func=cmd_gen)

    for name, fn in [("analyze", cmd_analyze), ("report", cmd_report)]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    sub.choices["report"].add_argument("--limit", type=int, default=10)

    lp = sub.add_parser("leap")
    lp.set_defaults(func=cmd_leap)

    s = sub.add_parser("scan")
    s.add_argument("dir", nargs="?", default="tests/corpus")
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--eval-corpus", default="tests/corpus")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("advise")
    a.add_argument("--limit", type=int, default=15)
    a.set_defaults(func=cmd_advise)

    e = sub.add_parser("eval")
    e.set_defaults(func=cmd_eval)

    sv = sub.add_parser("serve")
    sv.add_argument("--port", type=int, default=5199)
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
