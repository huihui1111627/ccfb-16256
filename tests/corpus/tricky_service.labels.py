# tricky_service.py 的人工标注真值 (行级)。
# bad: 应报的行 -> 类别
bad_lines = {
    9:  "sub_now",          # s1: _t.time() - start (别名)
    15: "sub_now",          # s2: time.time() - t (变量)
    19: "sub_now",          # s3: time.time() - created_at (参数)
    29: "timeout_wall",     # s5: if time.time() > deadline
    37: "sub_now",          # s6: datetime.now() 相减
    48: "timeout_wall",     # s8: if now >= entry.ttl_at
    60: "timestamp_for_id", # s10: 墙钟雪花 ID
    69: "deadline_wall",    # s12: lease_expire = time.time()+30
    70: "timeout_wall",     # s12: while time.time() < lease_expire
    86: "sub_now",          # s15: time.time() - t0
}
# 正确写法/无害用法, 绝不能报
good_lines = {23, 24, 25, 41, 42, 43, 53, 54, 55, 56, 65, 75, 76, 77, 80, 81, 82}
