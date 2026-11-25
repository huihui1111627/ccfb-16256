"""更刁钻的时间用法 —— 覆盖别名、间接、负例。用于测量扫描器的漏报/误报。"""
import time as _t
import datetime
from datetime import datetime
import functools


def s1(start):                                  # BAD: 别名 time as _t
    return _t.time() - start


def s2():                                       # BAD: 先存 now 再相减 (变量)
    t = time.time()
    do()
    return time.time() - t


def s3(created_at):                             # BAD: 参数名 created_at 参与减法
    return time.time() - created_at


def s4():                                       # GOOD: 单调
    t = time.monotonic()
    do()
    return time.monotonic() - t


def s5(deadline):                               # BAD: 与 deadline 比较
    if time.time() > deadline:
        return False
    return True


def s6():                                       # BAD: datetime.now() 相减
    a = datetime.now()
    do()
    return (datetime.now() - a).total_seconds()


def s7(xs):                                     # OK: 只解析/格式化, 不做差
    stamp = datetime.now().isoformat()
    print(stamp)
    return xs


def s8(entry, now=None):                        # BAD: now 默认参数取墙钟
    now = now or time.time()
    if now >= entry.ttl_at:
        return None


def s9():                                       # GOOD: 用 perf_counter 做超时
    t = time.perf_counter()
    do()
    if time.perf_counter() - t > 1.0:
        raise TimeoutError


def s10():                                      # BAD: 墙钟生成雪花 ID 一部分
    nid = (int(datetime.now().timestamp() * 1000) << 12) | seq()
    return nid


def s11():                                      # GOOD: 随机/哈希, 不涉时间
    return hash((user_id(), counter()))


def s12():                                      # BAD: 租约到期用墙钟比较
    lease_expire = time.time() + 30
    while time.time() < lease_expire:
        try_renew()


def s13(records):                               # LOW/OK: 按日期过滤 (不算时长)
    day = datetime.now().date()
    return [r for r in records if r.date() == day]


def s14():                                      # GOOD: 单调比较
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        spin()


def s15(t0):                                    # BAD: 参数 t0 与墙钟相减
    return int((time.time() - t0) * 1000)


def do():
    pass


def seq():
    return 1


def user_id():
    return 1


def counter():
    return 1


def try_renew():
    pass


def spin():
    pass
