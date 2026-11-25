"""订单服务 —— 含多种时间用法, 部分正确部分危险。用作扫描器评测语料。"""
import time
import datetime
from datetime import datetime as dt
from time import time as now_ts, monotonic


def measure_elapsed_wall(start):              # BAD: 传入墙钟, 相减算耗时
    elapsed = time.time() - start
    return elapsed


def measure_elapsed_mono():                   # GOOD: 单调时钟
    t0 = time.monotonic()
    do_work()
    return time.monotonic() - t0


def acquire_lock_wall(redis, key, ttl):       # BAD: 租约用墙钟
    now = time.time()
    deadline = now + ttl
    ok = redis.set(key, "1", nx=True)
    while time.time() < deadline:
        if not ok:
            break
    return ok


def handle_request(req):                      # BAD: 超时用墙钟
    start = time.time()
    resp = call_upstream(req)
    if time.time() - start > req.timeout:
        raise TimeoutError
    return resp


def handle_request_good(req):                 # GOOD: 单调时钟超时
    start = time.monotonic()
    resp = call_upstream(req)
    if time.monotonic() - start > req.timeout:
        raise TimeoutError
    return resp


def gen_order_id():                           # BAD: 墙钟生成 ID
    order_id = int(time.time() * 1000)
    return order_id


def gen_order_id_str():                       # BAD: 墙钟拼字符串 ID
    order_id = f"{dt.now():%Y%m%d%H%M%S%f}"
    return order_id


def gen_seq_id(counter):                      # GOOD: 不是墙钟
    seq = counter.increment()
    return seq


def log_event(msg):                           # OK: 仅记录时间戳, 不做差
    ts = time.time()
    logger.info("%s %s", ts, msg)


def poll_until_done_wall():                   # BAD: 墙钟轮询
    start = time.time()
    while time.time() - start < 30:
        if check_done():
            return True
        time.sleep(0.1)
    return False


def poll_until_done_mono():                   # GOOD: 单调轮询
    start = time.monotonic()
    while time.monotonic() - start < 30:
        if check_done():
            return True
        time.sleep(0.1)
    return False


def bucket_by_day(rows):                      # LOW: 按日期分桶 (闰秒影响小)
    today = dt.now().date()
    return [r for r in rows if r.created.date() == today]


def cache_ttl_check(entry):                   # BAD: TTL 用墙钟
    if time.time() > entry.expires_at:
        return None
    return entry.value


def cache_ttl_check_mono(entry):              # GOOD
    if monotonic() > entry.mono_expires:
        return None
    return entry.value


def date_diff_days(a, b):                     # BAD: 墙钟相减 (但跨天)
    delta = a - b
    return delta.days


def do_work():
    return 1


def call_upstream(req):
    return req


def check_done():
    return True


class Logger:
    def info(self, *a):
        pass


logger = Logger()
