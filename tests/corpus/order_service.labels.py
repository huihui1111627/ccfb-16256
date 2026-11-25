# 扫描语料的人工标注真值: 每行一个 "应报" 的 (行号, 类别)。
# 类别用于评估召回; 扫描器报出的行号只要落在这些行上即算命中。
# GOOD / OK 示例 (单调时钟 / 仅记录时间戳) 不得报出 —— 用于评估误报。
# 行号对应当前 tests/corpus/order_service.py。

bad_lines = {
    9:  "sub_now",          # measure_elapsed_wall: time.time() - start
    21: "deadline_wall",    # acquire_lock_wall: deadline = now + ttl
    23: "timeout_wall",     # while time.time() < deadline  (租约)
    32: "timeout_wall",     # handle_request: if time.time()-start > timeout
    46: "timestamp_for_id", # gen_order_id: int(time.time()*1000)
    51: "timestamp_for_id", # gen_order_id_str: f"{dt.now():...}"
    67: "sleep_loop_wall",  # poll_until_done_wall: while time.time()-start < 30
    89: "timeout_wall",     # cache_ttl_check: if time.time() > entry.expires_at
}

# 绝不应报告的行 (正确写法 / 无害用法)
good_lines = {16, 17, 18, 19, 20, 39, 40, 41, 42, 43, 44, 55, 56, 57, 61, 75, 76, 77, 78, 79, 80, 82, 83, 95, 96, 97}

# 扫描器在同一行可能报多个类别 (如 23 行既 timeout 又 sleep_loop) —— 行级评估时视为命中。
