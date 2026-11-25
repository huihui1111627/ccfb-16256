"""业务代码墙上时钟扫描 (AST).

目标: 找出 **用墙上时间来算时长和超时** 的地方 —— 这些在时间跳变/闰秒时最容易出事,
以及用墙钟生成编号、做租约、拼日期分桶等模式。

与 "正确做法" 的对照:
  * 算时长/超时/耗时 -> 应该用**单调时钟**: `time.monotonic()`, `time.perf_counter()`
    (Go: time.Since 在 Go1.9+ 已含单调读数; 但 `time.Now().Sub()` 存成 timestamp 再用会退化)
  * 排序/审计/跨机比对 -> 可以用墙钟, 但不应做差值
  * 生成编号 -> 不能用墙钟往回走

检测项 (每项给出 file:line, 函数, 严重度, 说明):
  1. sub_now             : `time.time() - saved` / `datetime.now() - start` 这种墙钟差
  2. elapsed_wall        : 变量名含 start/begin/t0 ... 且用 time.time()/now() 记录
  3. timeout_wall        : deadline / timeout / expires 用墙钟计算
  4. sleep_loop_wall     : 用墙钟轮询判断 "过了多久"
  5. wall_increase       : 同函数内先 time.time() 再相减 -> 跨度测量
  6. timestamp_for_id    : 生成 id / seq 时用 int(time.time()|now)
  7. lease_wall          : lease/ttl/lock 相关变量使用墙钟
  8. date_bucket         : 用本地日期做分桶/文件名 (闰秒影响小, 但跨时区/跳变需注意)

为了减少误报, 做函数级数据流: 追踪 "墙钟来源变量" 在该函数内的使用。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

WALL_FUNCS = {
    "time.time", "datetime.now", "datetime.utcnow", "datetime.today",
    "datetime.datetime.now", "datetime.datetime.utcnow", "datetime.datetime.today",
    "time.time_ns",
}
MONO_FUNCS = {
    "time.monotonic", "time.monotonic_ns", "time.perf_counter", "time.perf_counter_ns",
    "time.process_time",
}

WALL_ATTR = {"time", "now", "utcnow", "today", "time_ns"}
TIME_NAMES = {"time", "datetime", "dt"}


@dataclass
class Finding:
    path: str
    lineno: int
    col: int
    func: str
    rule: str
    severity: str           # low | medium | high | critical
    message: str
    snippet: str
    suggestion: str

    def to_dict(self) -> dict:
        return {
            "path": self.path, "lineno": self.lineno, "col": self.col,
            "func": self.func, "rule": self.rule, "severity": self.severity,
            "message": self.message, "snippet": self.snippet,
            "suggestion": self.suggestion,
        }


SEV_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class FuncVisitor(ast.NodeVisitor):
    """在一个函数体内做轻量数据流: 追踪墙钟来源变量 -> 相减 -> 用于时长/超时。"""

    def __init__(self, path: str, src: str, imports: Dict[str, str]):
        self.path = path
        self.src = src
        self.lines = src.splitlines()
        self.imports = imports
        self.findings: List[Finding] = []
        self.func_stack: List[str] = []
        self.wall_vars: Set[str] = set()
        self.mono_vars: Set[str] = set()
        self.assigned_at: Dict[str, int] = {}

    # ---------- helpers
    def _fqn(self, node: ast.AST) -> Optional[str]:
        """把调用目标解析为点号全名 (尽力而为, 处理 import as)。"""
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            dotted = ".".join(parts)
            # import 别名替换
            head = parts[0]
            if head in self.imports:
                rest = ".".join(parts[1:])
                base = self.imports[head]
                return base + ("." + rest if rest else "")
            return dotted
        return None

    def _is_wall(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            # datetime.now() 本身
            if self._is_wall(node.func):
                return True
            node = node.func
        # attr 链的基座可能是另一个 Call (datetime.now().timestamp())
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call):
            return self._is_wall(node.value)
        fq = self._fqn(node)
        if fq is None:
            return False
        if fq in WALL_FUNCS:
            return True
        # from time import time; time()
        if fq in ("time",) and self.imports.get("time") == "time.time":
            return True
        last2 = ".".join(fq.split(".")[-2:])
        return last2 in {"datetime.now", "datetime.utcnow", "datetime.today"}

    def _is_mono(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            node = node.func
        fq = self._fqn(node)
        return fq in MONO_FUNCS

    def _snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()[:160]
        return ""

    def _add(self, node: ast.AST, rule: str, sev: str, msg: str, sug: str):
        fn = self.func_stack[-1] if self.func_stack else "<module>"
        self.findings.append(Finding(
            path=self.path, lineno=getattr(node, "lineno", 0),
            col=getattr(node, "col_offset", 0), func=fn, rule=rule,
            severity=sev, message=msg, snippet=self._snippet(getattr(node, "lineno", 0)),
            suggestion=sug))

    # ---------- visitors
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_func(node)

    def _visit_func(self, node):
        # 每函数独立的墙钟变量集合
        saved_wall, saved_mono = self.wall_vars, self.mono_vars
        saved_at = self.assigned_at
        self.wall_vars, self.mono_vars = set(), set()
        self.assigned_at = {}
        self.func_stack.append(node.name)
        # 参数名启发式: 形参若是 start/begin/t0/since/created_at/now_ts 等,
        # 且函数内与墙钟做减法, 很可能是传入的墙钟时间戳。
        self._param_wall_names = {
            a.arg for a in node.args.args + node.args.kwonlyargs
            if any(k in a.arg.lower() for k in
                   ("start", "begin", "t0", "since", "created", "begin_ts", "start_time"))
        }
        # 预扫描: 先把所有 "x = 墙钟" 的变量收进集合, 避免顺序依赖
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                val = getattr(sub, "value", None)
                if val is None:
                    continue
                if self._expr_is_wall(val):
                    for tgt in (sub.targets if isinstance(sub, ast.Assign) else [sub.target]):
                        if isinstance(tgt, ast.Name):
                            self.wall_vars.add(tgt.id)
                elif self._expr_is_mono(val):
                    for tgt in (sub.targets if isinstance(sub, ast.Assign) else [sub.target]):
                        if isinstance(tgt, ast.Name):
                            self.mono_vars.add(tgt.id)
        for stmt in node.body:
            self.visit(stmt)
        self.func_stack.pop()
        self.wall_vars, self.mono_vars = saved_wall, saved_mono
        self.assigned_at = saved_at
        self._param_wall_names = set()

    def visit_Assign(self, node: ast.Assign):
        wall = self._expr_is_wall(node.value)
        mono = self._expr_is_mono(node.value)
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                if wall:
                    self.wall_vars.add(tgt.id)
                    self.assigned_at[tgt.id] = node.lineno
                elif mono:
                    self.mono_vars.add(tgt.id)
        # id 生成: 赋值目标是 id/seq/uid + 值是 int(墙钟)
        self._check_id_assign(node)
        # deadline = wall_now + ttl  (把墙钟写进到期时刻)
        tgt_names = [t.id.lower() for t in node.targets if isinstance(t, ast.Name)]
        if any(any(k in t for k in ("deadline", "expire", "expires_at", "due", "timeout_at"))
               for t in tgt_names):
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                if self._expr_is_wall(node.value.left) or self._expr_is_wall(node.value.right):
                    self._add(node, "deadline_wall", "critical",
                              "用墙钟计算到期时刻 (deadline = now + ttl)",
                              "到期时刻应基于单调时钟; 墙钟倒退会让 deadline 判失效, 提前/延后过期")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None and isinstance(node.target, ast.Name):
            if self._expr_is_wall(node.value):
                self.wall_vars.add(node.target.id)
            elif self._expr_is_mono(node.value):
                self.mono_vars.add(node.target.id)
        self.generic_visit(node)

    def _expr_is_wall(self, e: ast.AST) -> bool:
        if isinstance(e, ast.Call):
            if self._is_wall(e) or self._is_wall(e.func):
                return True
            # int(time.time()), float(...), round(...)
            for a in e.args:
                if self._expr_is_wall(a):
                    return True
        if isinstance(e, ast.Name) and e.id in self.wall_vars:
            return True
        if isinstance(e, ast.BinOp):
            return self._expr_is_wall(e.left) or self._expr_is_wall(e.right)
        if isinstance(e, ast.BoolOp):
            return any(self._expr_is_wall(v) for v in e.values)
        if isinstance(e, ast.IfExp):
            return self._expr_is_wall(e.body) or self._expr_is_wall(e.orelse)
        if isinstance(e, ast.Attribute):
            # datetime.now().timestamp() / datetime.now().date() / .total_seconds()
            if isinstance(e.value, ast.Call):
                return self._is_wall(e.value) or self._expr_is_wall(e.value)
            if isinstance(e.value, ast.Attribute):
                return self._expr_is_wall(e.value)
            return False
        return False

    def _expr_is_mono(self, e: ast.AST) -> bool:
        if isinstance(e, ast.Call):
            if self._is_mono(e) or self._is_mono(e.func):
                return True
            for a in e.args:
                if self._expr_is_mono(a):
                    return True
        if isinstance(e, ast.BinOp):
            return self._expr_is_mono(e.left) or self._expr_is_mono(e.right)
        return False

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.Sub):
            # now - wall_var  (或 wall_var - now)
            if self._is_wall(node.left) and self._name_is_wall(node.right):
                self._add(node, "sub_now", "critical",
                          "用当前墙钟减去之前记录的墙钟变量来算时长",
                          "改用 time.monotonic()/perf_counter() 计算 elapsed, 墙钟跳变会让差值失真甚至为负")
            elif self._is_wall(node.right) and self._name_is_wall(node.left):
                self._add(node, "sub_now", "critical",
                          "两个墙钟时间点相减算时长",
                          "改用单调时钟; 跨闰秒 step 会使结果为负 (超时/租约直接失效)")
            elif self._name_is_wall(node.left) and self._name_is_wall(node.right):
                self._add(node, "wall_diff", "high",
                          "两个墙钟变量相减",
                          "若用于时长/超时判断, 改用单调时钟")
            # 形参 start/begin 与墙钟相减
            elif (self._is_wall(node.left) and self._is_param_wall(node.right)) or \
                 (self._is_wall(node.right) and self._is_param_wall(node.left)):
                self._add(node, "sub_now", "high",
                          "墙钟减去形参 (疑似传入的起始时间戳) 算耗时",
                          "若 start 是墙钟, 应改为在调用方用单调时钟取基准; 墙钟跳变会让耗时为负")
        self.generic_visit(node)

    def _is_param_wall(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Name)
                and node.id in getattr(self, "_param_wall_names", set()))

    def _name_is_wall(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.wall_vars

    def _check_deadline(self, node: ast.BinOp):
        # 形如 x = wall + something, 且 x 名字像 deadline/timeout/expire
        pass

    def visit_Compare(self, node: ast.Compare):
        # if time.time() > deadline / if now > entry.ttl_at
        has_wall = any(self._is_wall(n) for n in ast.walk(node))
        has_wall_var = any(isinstance(n, ast.Name) and n.id in self.wall_vars for n in ast.walk(node))
        attr_names = []
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute):
                attr_names.append(n.attr.lower())
            elif isinstance(n, ast.Name):
                attr_names.append(n.id.lower())
        joined = " ".join(attr_names)
        if (has_wall or has_wall_var) and any(k in joined for k in
                            ("timeout", "deadline", "expire", "ttl", "lease", "elapsed", "due")):
            self._add(node, "timeout_wall", "high",
                      "用墙钟与到期时刻/超时阈值比较",
                      "超时/租约判断应基于单调时钟; 墙钟倒退会让已到期判为未到期")
        self.generic_visit(node)

    def _name_of(self, n):
        return getattr(n, "id", "")

    def visit_While(self, node: ast.While):
        has_wall = any(self._is_wall(n) for n in ast.walk(node.test))
        has_wallvar = any(isinstance(n, ast.Name) and n.id in self.wall_vars for n in ast.walk(node.test))
        if has_wall or has_wallvar:
            self._add(node, "sleep_loop_wall", "high",
                      "用墙钟轮询判断 '过了多久' 的循环",
                      "改用单调时钟; 墙钟跳变会让轮询提前/永不结束")
        self.generic_visit(node)

    def _check_id_assign(self, node: ast.Assign):
        depth = 0
        tgt_names = [t.id.lower() for t in node.targets if isinstance(t, ast.Name)]
        if not any(any(k in t for k in ("id", "seq", "uid", "token", "snowflake", "orderno", "order_no"))
                   for t in tgt_names):
            return
        if self._expr_is_wall(node.value):
            self._add(node, "timestamp_for_id", "critical",
                      "用墙钟时间戳生成 ID/序列号",
                      "时钟倒退会生成更小或重复的 ID; 需时钟回拨保护或改用单调序列")
        # id = f"{now:%Y%m%d}..." 也归类
        if isinstance(node.value, ast.JoinedStr):
            for v in node.value.values:
                if isinstance(v, ast.FormattedValue) and self._expr_is_wall(v.value):
                    self._add(node, "timestamp_for_id", "critical",
                              "用墙钟格式化字符串拼 ID",
                              "按时间分桶的 ID 在跳变/闰秒时会重复或乱序")
                    break

    def visit_Call(self, node: ast.Call):
        fq = self._fqn(node.func) or ""
        # time.sleep / datetime.now 在参数里
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        self.generic_visit(node)


# ---------------------------------------------------------------- 模块级

def scan_source(path: str, src: str) -> List[Finding]:
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return [Finding(path=path, lineno=e.lineno or 0, col=0, func="<syntax>",
                        rule="parse_error", severity="low",
                        message=f"无法解析: {e.msg}", snippet="", suggestion="")]
    imports = _collect_imports(tree)
    v = FuncVisitor(path, src, imports)
    v.visit(tree)
    return _dedupe(v.findings)


def _collect_imports(tree: ast.AST) -> Dict[str, str]:
    """别名 -> 真实全名, 用于解析 time/datetime 调用。"""
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    out[a.name.split(".")[0]] = a.name.split(".")[0]
                    # import time -> time 名指向 time 模块
                if a.name == "time":
                    out["time"] = "time"
                elif a.name == "datetime":
                    out["datetime"] = "datetime"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                if a.asname:
                    out[a.asname] = f"{mod}.{a.name}"
                else:
                    out[a.name] = f"{mod}.{a.name}"
    return out


def _dedupe(findings: List[Finding]) -> List[Finding]:
    seen = set()
    out = []
    for f in findings:
        key = (f.path, f.lineno, f.rule)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def scan_file(path: str) -> List[Finding]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    return scan_source(path, src)


def scan_dir(root: str, exts: Tuple[str, ...] = (".py",),
             exclude: Iterable[str] = ("__pycache__", ".venv", "venv", "node_modules", ".git")) -> List[Finding]:
    out: List[Finding] = []
    excl = set(exclude)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excl]
        for fn in filenames:
            if fn.endswith(exts):
                out.extend(scan_file(os.path.join(dirpath, fn)))
    return out


def summarize(findings: List[Finding]) -> dict:
    by_rule: Dict[str, int] = {}
    by_sev: Dict[str, int] = {}
    for f in findings:
        by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
    return {
        "total": len(findings),
        "by_rule": dict(sorted(by_rule.items(), key=lambda x: -x[1])),
        "by_severity": dict(sorted(by_sev.items(), key=lambda x: -SEV_RANK.get(x[0], 0))),
        "findings": [f.to_dict() for f in findings],
    }
