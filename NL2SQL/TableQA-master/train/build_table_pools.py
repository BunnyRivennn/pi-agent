# -*- coding: utf-8 -*-
"""
构建 NL2SQL 合成数据的表池（P0~P6）与表能力画像。

输入:
    database_inspection.json   5013 张表的 schema 与列统计
    train.tables.json          每表的业务表头(header)/标题(title)，按列顺序对齐 col_1..col_N
输出:
    table_pools.json           表池成员 + 每表能力画像，供配额调度直接读取

判定原则:
    1. 聚合能力用"业务名 + 数值统计"共同判定，不单看类型。编号/序号/年份等不允许 SUM/AVG。
    2. 分组列要求 distinct 适中且平均组内行数够，否则分组后每组一行，聚合无意义。
    3. 日期列按可比较格式归类，格式不可靠的不参与时间任务。
"""
import json
import re
from collections import Counter, defaultdict

INSPECT = "database_inspection.json"
TABLES = "train.tables.json"
OUT = "table_pools.json"

# ---------- 业务名语义规则 ----------
# 命中则禁止 SUM/AVG（可以 COUNT、可以分组、可以排序）
ID_LIKE = re.compile(r"(序号|编号|代码|编码|代号|id$|ID|排名|名次|年份|年度|月份|日期|时间|电话|邮编|身份证|证号|批号|期号)")
# 命中则禁止 SUM（可以 AVG），比率类相加无意义
RATE_LIKE = re.compile(r"(率|占比|比例|百分比|同比|环比|增幅|涨幅|跌幅|%|％|评分|得分|指数|均价|平均)")
# 明确可加总
SUMMABLE = re.compile(r"(金额|收入|支出|成本|利润|产量|销量|数量|人数|户数|套数|面积|总额|总量|总计|合计|营业额|市值|资产|负债|投资|出让金|票房|客流|吞吐)")

# 实测库内日期格式很脏: 2006.8 / 18/10/31 / 1976年1月20日 / -- / REAL存的2018.0 都有。
# 只认两类可靠口径:
#   (a) 可按字符串直接比较且零填充一致的格式
#   (b) 纯年份数值列(按数值比较)
DATE_FORMATS = [
    (re.compile(r"^(19|20)\d{2}-\d{1,2}-\d{1,2}$"), "YYYY-MM-DD"),
    (re.compile(r"^(19|20)\d{2}/\d{1,2}/\d{1,2}$"), "YYYY/MM/DD"),
    (re.compile(r"^(19|20)\d{2}-\d{1,2}$"), "YYYY-MM"),
    (re.compile(r"^(19|20)\d{6}$"), "YYYYMMDD"),
    (re.compile(r"^(19|20)\d{2}年\d{1,2}月\d{1,2}日$"), "YYYY年MM月DD日"),
    (re.compile(r"^(19|20)\d{2}年\d{1,2}月$"), "YYYY年MM月"),
]
# 能在 SQLite 里按字符串比较出正确先后顺序的格式。
# 注意 YYYY/M/D、YYYY-M-D 这类月日不零填充的, 字符串比较会把 "2016/9/1" 排到 "2016/10/1" 之后,
# 因此只允许做等值/LIKE 前缀匹配, 不允许做 BETWEEN 范围查询。
COMPARABLE = {"YYYY-MM-DD", "YYYYMMDD", "YYYY-MM"}
NOISE = {None, "", "None", "--", "-", "/", "null", "NULL", "nan"}


def load_meta():
    meta = {}
    with open(TABLES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                meta[o["name"]] = o
    return meta


def date_format_of(samples):
    """返回 (格式名, 命中率)。要求样例值高度一致才认定为日期列。"""
    vals = [str(v) for v in (samples or []) if v not in NOISE]
    if len(vals) < 3:
        return None, 0.0
    best, best_hit = None, 0
    for pat, name in DATE_FORMATS:
        hit = sum(1 for v in vals if pat.match(v))
        if hit > best_hit:
            best, best_hit = name, hit
    return (best, best_hit / len(vals)) if best else (None, 0.0)


def is_year_column(cs):
    """纯年份数值列: 全部落在 1900~2100 的整数, 可按数值做范围比较。"""
    ns = cs.get("numeric_stats")
    if not ns:
        return False
    lo, hi = ns.get("min"), ns.get("max")
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return False
    if not (1900 <= lo and hi <= 2100):
        return False
    nums = [v for v in (cs.get("sample_values") or []) if isinstance(v, (int, float))]
    if not nums or not all(float(v) == int(float(v)) for v in nums):
        return False
    return cs["distinct_count"] >= 2


def profile_table(t, m):
    """生成单表能力画像。"""
    rc = t["row_count"]
    header = m["header"]
    cols = t["columns"]
    # col_N 与 header 按顺序对齐（已验证 5013 张表列数全部一致）
    biz = {c["name"]: (header[i] if i < len(header) else None) for i, c in enumerate(cols)}

    def enough(cs):
        return cs["non_null_count"] >= max(3, rc * 0.5)

    numeric, aggregatable, summable, groupable, date_cols, text_cols = [], [], [], [], [], []

    for cs in t["column_stats"]:
        name = cs["column_name"]
        bn = biz.get(name) or ""
        ns = cs.get("numeric_stats")
        dc = cs["distinct_count"]

        # --- 数值/聚合 ---
        if ns and enough(cs) and dc >= 3 and ns["max"] != ns["min"]:
            item = {"name": name, "business_name": bn,
                    "min": ns["min"], "max": ns["max"], "avg": ns["avg"], "distinct": dc}
            numeric.append(item)          # 任何有区分度的数值列都可用于筛选/比较
            # 编号判定以业务名为主。表头 5013 张全部可用，不再用 distinct==行数 这种启发式，
            # 否则小表里"金额"这类取值天然全不同的列会被误判成编号。
            is_id = bool(ID_LIKE.search(bn)) or (not bn and dc == rc)
            is_rate = bool(RATE_LIKE.search(bn))
            is_sum = bool(SUMMABLE.search(bn))
            if not is_id:
                aggregatable.append(item)                 # MIN / MAX / AVG 可用
                if is_sum and not is_rate:
                    summable.append(item)                 # SUM 仅限明确可加总且非比率

        # --- 日期 ---
        if is_year_column(cs) and enough(cs):
            date_cols.append({"name": name, "business_name": bn, "format": "YYYY_NUMERIC",
                              "comparable": True, "range_safe": True, "distinct": dc,
                              "min": cs["numeric_stats"]["min"], "max": cs["numeric_stats"]["max"]})
            continue
        fmt, hit = date_format_of(cs.get("sample_values"))
        if fmt and hit >= 0.8 and enough(cs):
            date_cols.append({"name": name, "business_name": bn, "format": fmt,
                              "comparable": True,
                              # 月日不零填充的格式字符串比较会错序, 只能等值/前缀匹配
                              "range_safe": fmt in COMPARABLE, "distinct": dc})
            continue  # 日期列不再作为普通文本/分组列

        # --- 文本/分组 ---
        if cs["declared_type"] == "TEXT" and enough(cs):
            if dc >= 3:
                text_cols.append({"name": name, "business_name": bn, "distinct": dc})
            avg_group = rc / dc if dc else 0
            if 2 <= dc < rc and dc <= min(30, max(2, rc * 0.6)):
                groupable.append({"name": name, "business_name": bn,
                                  "distinct": dc, "avg_group_size": round(avg_group, 2),
                                  "having_friendly": 3 <= dc <= 15 and avg_group >= 3})

    gb_having = [g for g in groupable if g["having_friendly"]]
    date_cmp = [d for d in date_cols if d["comparable"]]
    date_range = [d for d in date_cols if d.get("range_safe")]

    prof = {
        "row_count": rc,
        "column_count": len(cols),
        "title": (m.get("title") or "").strip(),
        "numeric_columns": numeric,
        "aggregatable_columns": aggregatable,
        "summable_columns": summable,
        "groupable_columns": groupable,
        "having_friendly_columns": gb_having,
        "date_columns": date_cols,
        "comparable_date_columns": date_cmp,
        "range_safe_date_columns": date_range,
        "text_columns": text_cols,
        "semantic_known": len(header) == len(cols),
    }

    # ---------- 表池归属 ----------
    # 关键: COUNT(*) 对任何表都成立, 所以分组类池(P2/P3)不要求有数值聚合列,
    #       只有需要 SUM/AVG/MIN/MAX 指标的池(P1/P4)才要求 aggregatable。
    pools = []
    n_num, n_agg = len(numeric), len(aggregatable)
    n_gb, n_gbh = len(groupable), len(gb_having)
    sortable = n_num >= 1 or len(date_cmp) >= 1          # 可排序: 数值或可比较日期

    if rc >= 5 and any(enough(cs) for cs in t["column_stats"]):
        pools.append("P0")                                # 基础筛选
    if rc >= 20 and sortable:
        pools.append("P1")                                # 比较/范围/排序/Top-K
    if rc >= 20 and n_agg >= 1:
        pools.append("P1A")                               # 需要真实数值聚合指标
    if rc >= 25 and n_gb >= 1:
        pools.append("P2")                                # GROUP BY (+COUNT 即可)
    if rc >= 30 and n_gbh >= 1:
        pools.append("P3")                                # GROUP BY + HAVING
    if rc >= 30 and n_gbh >= 1 and n_agg >= 1:
        pools.append("P4")                                # 多聚合指标 + HAVING + 多子句
    if rc >= 20 and len(date_cmp) >= 1:
        pools.append("P5")                                # 时间范围 + 分组/聚合
    if rc >= 10 and len(date_cmp) >= 1:
        pools.append("P5L")                               # 放宽版, 供 Easy 简单时间条件
    if len(text_cols) >= 1:
        pools.append("P6")                                # LIKE / IN / DISTINCT
    if rc >= 25 and n_gb >= 2:
        pools.append("PMG")                               # 多字段分组
    prof["pools"] = pools
    return prof


def main():
    meta = load_meta()
    insp = json.load(open(INSPECT, encoding="utf-8"))
    profiles, pools = {}, defaultdict(list)
    skipped = 0

    for t in insp["tables"]:
        m = meta.get(t["table_name"])
        if not m:
            skipped += 1
            continue
        p = profile_table(t, m)
        profiles[t["table_name"]] = p
        for pool in p["pools"]:
            pools[pool].append(t["table_name"])

    out = {
        "source": {"inspection": INSPECT, "tables_meta": TABLES},
        "total_tables": len(profiles),
        "skipped_no_meta": skipped,
        "pool_sizes": {k: len(v) for k, v in sorted(pools.items())},
        "pools": {k: v for k, v in sorted(pools.items())},
        "table_profiles": profiles,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"总表数 {len(profiles)}，无元信息跳过 {skipped}")
    print("--- 表池规模 ---")
    for k, v in out["pool_sizes"].items():
        print(f"{k:5s} {v:5d}")
    print("--- 能力统计 ---")
    c = Counter()
    for p in profiles.values():
        c["数值列>=1"] += len(p["numeric_columns"]) >= 1
        c["可聚合列>=1"] += len(p["aggregatable_columns"]) >= 1
        c["可SUM列>=1"] += len(p["summable_columns"]) >= 1
        c["HAVING友好列>=1"] += len(p["having_friendly_columns"]) >= 1
        c["可比较日期列>=1"] += len(p["comparable_date_columns"]) >= 1
    for k, v in c.items():
        print(f"{k} {v}")


if __name__ == "__main__":
    main()
