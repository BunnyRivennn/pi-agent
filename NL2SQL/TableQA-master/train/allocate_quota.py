# -*- coding: utf-8 -*-
"""
把 30,000 条配额分配到具体的表上，产出合成工单。

输入: table_pools.json  (由 build_table_pools.py 生成)
输出:
    synthesis_tasks.jsonl   工单: 每行 = (表, 难度, 任务, 该表该任务要生成几条)
    quota_report.json       配额完成情况与密度核查

分配规则:
    1. 每个任务只在其表池内选表。
    2. 同一张表同一任务 <= PER_TASK_CAP 条。
    3. 同一张表合计 <= PER_TABLE_CAP 条。
    4. 优先选"已用样本数最少"的表, 保证覆盖均匀、避免扎堆。
    5. 配额凑不满时记录告警, 不提高密度硬凑。
"""
import json
from collections import defaultdict

POOLS = "table_pools.json"
OUT_TASKS = "synthesis_tasks.jsonl"
OUT_REPORT = "quota_report.json"

PER_TASK_CAP = 2      # 同表同任务上限
# 同表合计上限。设 8 时 P1 池的表会被 P2/P3/P4 任务先占满, 导致 multi_cond 等任务缺 900 余条;
# 设 10 恰好能满额, 且实测最大值就是 10(设 12 也不会更高), 故定为 10。
# 参照: 原始 41522 条数据每表中位数 6 条、最多 51 条, 本配额均摊 6.02 条属同量级。
PER_TABLE_CAP = 10    # 同表合计上限

# (难度, 任务key, 中文名, 表池, 配额)
QUOTA = [
    # ---------------- Easy 12,000 ----------------
    ("easy", "where_single",      "基础 WHERE 单条件",            "P0",  2000),
    ("easy", "where_and",         "WHERE + AND",                 "P0",  1500),
    ("easy", "where_or",          "WHERE + OR",                  "P0",  1000),
    ("easy", "in_like_between",   "IN / LIKE / BETWEEN / NOT",   "P6",  1500),
    ("easy", "time_simple",       "简单时间条件",                  "P5L",  500),
    ("easy", "agg_simple",        "简单聚合 COUNT/SUM/AVG/MIN/MAX", "P1A", 1400),
    ("easy", "order_by",          "ORDER BY 单字段升降序",          "P1",  1000),
    ("easy", "order_limit",       "ORDER BY + LIMIT / Top-K",     "P1",  1100),
    ("easy", "distinct",          "DISTINCT",                     "P6",   500),
    ("easy", "combo_simple",      "简单组合 WHERE+ORDER BY/聚合",   "P1",  1500),
    # ---------------- Medium 12,000 ----------------
    ("medium", "multi_cond",      "多条件 AND/OR 组合(含括号)",     "P1",  1500),
    ("medium", "agg_multi",       "聚合函数(多指标, 带 WHERE)",     "P1A", 1500),
    ("medium", "groupby_agg",     "GROUP BY + 聚合",              "P2",  2200),
    ("medium", "where_groupby",   "WHERE + GROUP BY",             "P2",  1200),
    ("medium", "groupby_having",  "GROUP BY + HAVING",            "P3",  1200),
    ("medium", "agg_order_limit", "ORDER BY + LIMIT (聚合结果)",   "P1A", 1000),
    ("medium", "agg_order",       "聚合结果排序",                  "P2",   800),
    ("medium", "distinct_agg",    "DISTINCT + 聚合",              "P6",   500),
    ("medium", "time_agg",        "时间条件 + 聚合/分组",           "P5",   400),
    ("medium", "combo_multi",     "多子句组合 W+G+O+L",            "P2",  1700),
    # ---------------- Hard 6,000 ----------------
    ("hard", "complex_logic",     "复杂 AND/OR 嵌套逻辑",          "P1",   800),
    ("hard", "groupby_multiagg_having", "GROUP BY + 多聚合 + HAVING", "P4", 1100),
    ("hard", "groupby_having_order",    "GROUP BY + HAVING + ORDER BY", "P3", 900),
    ("hard", "full_combo",        "W+G+H+O+L 五子句组合",          "P4",  1000),
    ("hard", "multi_groupby",     "多字段 GROUP BY + 多聚合",       "PMG",  800),
    ("hard", "time_complex",      "时间 + 聚合 + 分组 + 排序",      "P5",   250),
    ("hard", "edge_semantic",     "复杂语义 / 边界场景",            "P2",   750),
    ("hard", "subquery",          "子查询(同表, 需显式开启)",        "P3",   400),
]

# 任务对表的额外要求: 池只保证基础能力, 这里做任务级精筛
TASK_REQUIRE = {
    "agg_simple":        lambda p: len(p["aggregatable_columns"]) >= 1,
    "agg_multi":         lambda p: len(p["aggregatable_columns"]) >= 1,
    "agg_order_limit":   lambda p: len(p["aggregatable_columns"]) >= 1,
    "time_simple":       lambda p: len(p["comparable_date_columns"]) >= 1,
    "time_agg":          lambda p: len(p["comparable_date_columns"]) >= 1 and len(p["groupable_columns"]) >= 1,
    "time_complex":      lambda p: len(p["range_safe_date_columns"]) >= 1 and len(p["having_friendly_columns"]) >= 1,
    "groupby_multiagg_having": lambda p: len(p["aggregatable_columns"]) >= 1 and len(p["having_friendly_columns"]) >= 1,
    "full_combo":        lambda p: len(p["aggregatable_columns"]) >= 1 and len(p["having_friendly_columns"]) >= 1,
    "multi_groupby":     lambda p: len(p["groupable_columns"]) >= 2,
    "subquery":          lambda p: len(p["aggregatable_columns"]) >= 1,
    "in_like_between":   lambda p: len(p["text_columns"]) >= 1 or len(p["numeric_columns"]) >= 1,
    "distinct_agg":      lambda p: len(p["text_columns"]) >= 1,
}


def allocate():
    data = json.load(open(POOLS, encoding="utf-8"))
    pools, profiles = data["pools"], data["table_profiles"]

    used_total = defaultdict(int)                  # 表 -> 已分配总数
    used_task = defaultdict(int)                   # (表, 任务) -> 已分配数
    orders = []
    report = {"quota": [], "warnings": []}

    # 按池大小升序处理: 稀缺池先占位, 宽松池后补, 避免稀缺任务没表可用
    ordered = sorted(QUOTA, key=lambda q: len(pools.get(q[3], [])))

    for diff, task, cname, pool, quota in ordered:
        cands = pools.get(pool, [])
        req = TASK_REQUIRE.get(task)
        if req:
            cands = [t for t in cands if req(profiles[t])]

        assigned, rounds = 0, 0
        # 多轮分配: 每轮给每张可用表加 1 条, 天然实现"先铺广度再加深度"
        while assigned < quota and rounds < PER_TASK_CAP:
            rounds += 1
            avail = [t for t in cands
                     if used_task[(t, task)] < rounds
                     and used_total[t] < PER_TABLE_CAP]
            # 本轮内按已用总数升序, 让冷门表先拿
            avail.sort(key=lambda t: (used_total[t], t))
            for t in avail:
                if assigned >= quota:
                    break
                used_task[(t, task)] += 1
                used_total[t] += 1
                assigned += 1

        # 汇总成工单: 同表同任务合并为一条, 带 count
        per_table = {}
        for t in cands:
            n = used_task[(t, task)]
            if n:
                per_table[t] = n
        for t, n in per_table.items():
            orders.append({"table_name": t, "difficulty": diff, "task": task,
                           "task_name_cn": cname, "pool": pool, "target_count": n})

        density = round(assigned / len(per_table), 2) if per_table else 0
        rec = {"difficulty": diff, "task": task, "task_name_cn": cname, "pool": pool,
               "pool_size": len(pools.get(pool, [])), "eligible_tables": len(cands),
               "quota": quota, "assigned": assigned,
               "tables_used": len(per_table), "density": density}
        report["quota"].append(rec)
        if assigned < quota:
            report["warnings"].append(
                f"[配额未满] {diff}/{task} 目标 {quota} 实配 {assigned}，"
                f"可用表 {len(cands)} 张，缺口 {quota - assigned}")

    # ---------- 核查 ----------
    report["quota"].sort(key=lambda r: (["easy", "medium", "hard"].index(r["difficulty"]), -r["quota"]))
    total = sum(r["assigned"] for r in report["quota"])
    by_diff = defaultdict(int)
    for r in report["quota"]:
        by_diff[r["difficulty"]] += r["assigned"]

    over_task = [(k, v) for k, v in used_task.items() if v > PER_TASK_CAP]
    over_table = [(k, v) for k, v in used_total.items() if v > PER_TABLE_CAP]
    report["summary"] = {
        "total_assigned": total,
        "by_difficulty": dict(by_diff),
        "tables_covered": len([t for t, n in used_total.items() if n > 0]),
        "total_tables": len(profiles),
        "max_per_table": max(used_total.values()) if used_total else 0,
        "avg_per_covered_table": round(total / max(1, len([t for t, n in used_total.items() if n > 0])), 2),
        "violations_per_task_cap": len(over_task),
        "violations_per_table_cap": len(over_table),
        "order_lines": len(orders),
    }

    with open(OUT_TASKS, "w", encoding="utf-8") as f:
        for o in orders:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---------- 打印 ----------
    s = report["summary"]
    print(f"总配额 {total}  难度分布 {dict(by_diff)}")
    print(f"覆盖表 {s['tables_covered']}/{s['total_tables']}  "
          f"单表最多 {s['max_per_table']} 条  覆盖表均摊 {s['avg_per_covered_table']} 条")
    print(f"密度违例: 同表同任务 {s['violations_per_task_cap']} 处, 同表合计 {s['violations_per_table_cap']} 处")
    print(f"工单行数 {s['order_lines']}  ->  {OUT_TASKS}")
    print("\n难度   任务                              池    可用表  配额   实配  用表   密度")
    for r in report["quota"]:
        flag = "" if r["assigned"] >= r["quota"] else "  <= 未满"
        print(f"{r['difficulty']:7s}{r['task_name_cn'][:26]:28s}{r['pool']:5s}"
              f"{r['eligible_tables']:7d}{r['quota']:7d}{r['assigned']:7d}"
              f"{r['tables_used']:6d}{r['density']:7.2f}{flag}")
    if report["warnings"]:
        print("\n--- 告警 ---")
        for w in report["warnings"]:
            print(w)


if __name__ == "__main__":
    allocate()
