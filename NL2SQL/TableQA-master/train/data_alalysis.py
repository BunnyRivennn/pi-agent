import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

AGGREGATE_FUNCS = {"COUNT", "SUM", "AVG", "MAX", "MIN"}

SQL_KEYWORDS = {
    "WHERE", "AND", "OR", "GROUP BY", "HAVING",
    "ORDER BY", "LIMIT", "DISTINCT", "IN", "LIKE",
    "BETWEEN", "NOT", "JOIN", "UNION", "WITH", "CASE"
}


def normalize_sql(sql: str) -> str:
    """
    SQL 归一化：
    - 转为大写
    - 字符串和数字字面量替换为 ?
    - 压缩空白字符

    用于初步统计模板重复率。
    """
    if not sql:
        return ""

    sql = sql.upper()

    # 替换字符串常量
    sql = re.sub(r"'(?:''|[^'])*'", "?", sql)

    # 替换数字常量
    sql = re.sub(r"\b\d+(?:\.\d+)?\b", "?", sql)

    # 压缩空白
    sql = re.sub(r"\s+", " ", sql).strip()

    return sql


def count_sql_features(sql: str) -> dict:
    """
    统计单条 SQL 的结构特征。
    """
    if not sql:
        return {}

    sql_upper = sql.upper()

    features = {}

    # 基础结构
    features["where"] = bool(re.search(r"\bWHERE\b", sql_upper))
    features["and_count"] = len(re.findall(r"\bAND\b", sql_upper))
    features["or_count"] = len(re.findall(r"\bOR\b", sql_upper))

    # 聚合函数
    agg_funcs = re.findall(
        r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(",
        sql_upper
    )
    features["aggregate_funcs"] = Counter(agg_funcs)

    # 其他 SQL 特征
    features["group_by"] = bool(
        re.search(r"\bGROUP\s+BY\b", sql_upper)
    )
    features["having"] = bool(re.search(r"\bHAVING\b", sql_upper))
    features["order_by"] = bool(
        re.search(r"\bORDER\s+BY\b", sql_upper)
    )
    features["limit"] = bool(re.search(r"\bLIMIT\b", sql_upper))
    features["distinct"] = bool(re.search(r"\bDISTINCT\b", sql_upper))
    features["in"] = bool(re.search(r"\bIN\s*\(", sql_upper))
    features["like"] = bool(re.search(r"\bLIKE\b", sql_upper))
    features["between"] = bool(re.search(r"\bBETWEEN\b", sql_upper))
    features["not"] = bool(re.search(r"\bNOT\b", sql_upper))
    features["join"] = bool(re.search(r"\bJOIN\b", sql_upper))
    features["union"] = bool(re.search(r"\bUNION\b", sql_upper))
    features["with"] = bool(re.search(r"\bWITH\b", sql_upper))
    features["case"] = bool(re.search(r"\bCASE\b", sql_upper))

    # 子查询：简单检测括号内是否包含 SELECT
    features["subquery"] = bool(
        re.search(r"\(\s*SELECT\b", sql_upper)
    )

    # WHERE 条件数量的近似统计
    where_match = re.search(
        r"\bWHERE\b(.*?)(?=\bGROUP\s+BY\b|\bHAVING\b|"
        r"\bORDER\s+BY\b|\bLIMIT\b|$)",
        sql_upper
    )

    if where_match:
        where_clause = where_match.group(1)
        and_count = len(re.findall(r"\bAND\b", where_clause))
        or_count = len(re.findall(r"\bOR\b", where_clause))
        features["where_condition_count"] = and_count + or_count + 1
    else:
        features["where_condition_count"] = 0

    # SELECT 字段数：初步估算
    select_match = re.search(
        r"\bSELECT\b(.*?)\bFROM\b",
        sql_upper
    )

    if select_match:
        select_clause = select_match.group(1)
        features["select_field_count"] = (
                select_clause.count(",") + 1
        )
    else:
        features["select_field_count"] = 0

    return features


def analyze_sql_dataset(jsonl_path: str) -> dict:
    """
    分析 JSONL 训练数据中的 SQL 结构覆盖情况。

    输入格式示例：
    {
        "question": "...",
        "table_id": "...",
        "sql": "SELECT ..."
    }
    """
    total = 0

    feature_counts = Counter()
    aggregate_counts = Counter()
    condition_counts = Counter()
    sql_pattern_counts = Counter()
    table_counts = Counter()

    feature_combinations = Counter()

    invalid_sql_count = 0
    missing_sql_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                invalid_sql_count += 1
                continue

            sql = item.get("sql", "")

            if not sql:
                missing_sql_count += 1
                continue

            total += 1

            table_id = item.get("table_id")
            if table_id:
                table_counts[table_id] += 1

            features = count_sql_features(sql)

            # 单项结构统计
            if features.get("where"):
                feature_counts["WHERE"] += 1

            if features.get("and_count", 0) > 0:
                feature_counts["AND"] += 1

            if features.get("or_count", 0) > 0:
                feature_counts["OR"] += 1

            for func, count in features.get(
                    "aggregate_funcs", {}
            ).items():
                aggregate_counts[func] += count

            for key in [
                "group_by", "having", "order_by", "limit",
                "distinct", "in", "like", "between", "not",
                "join", "subquery", "union", "with", "case"
            ]:
                if features.get(key):
                    feature_counts[key.upper()] += 1

            condition_counts[
                features.get("where_condition_count", 0)
            ] += 1

            # SQL 结构组合统计
            combination = tuple(
                name
                for name, enabled in [
                    ("WHERE", features.get("where")),
                    ("AGGREGATE", bool(
                        features.get("aggregate_funcs")
                    )),
                    ("GROUP_BY", features.get("group_by")),
                    ("HAVING", features.get("having")),
                    ("ORDER_BY", features.get("order_by")),
                    ("LIMIT", features.get("limit")),
                    ("DISTINCT", features.get("distinct")),
                ]
                if enabled
            )

            feature_combinations[combination] += 1

            # SQL 模板重复统计
            normalized = normalize_sql(sql)
            sql_pattern_counts[normalized] += 1

    unique_patterns = len(sql_pattern_counts)
    duplicate_excess = total - unique_patterns

    result = {
        "total_samples": total,
        "invalid_json_lines": invalid_sql_count,
        "missing_sql_samples": missing_sql_count,
        "feature_counts": feature_counts,
        "aggregate_counts": aggregate_counts,
        "where_condition_distribution": condition_counts,
        "feature_combinations": feature_combinations,
        "unique_sql_patterns": unique_patterns,
        "duplicate_excess": duplicate_excess,
        "duplicate_rate": (
            duplicate_excess / total if total else 0
        ),
        "table_count": len(table_counts),
        "top_tables": table_counts.most_common(20),
        "top_sql_patterns": sql_pattern_counts.most_common(20),
    }

    return result


def print_analysis_report(result: dict):
    total = result["total_samples"]

    print("=" * 60)
    print("SQL 数据结构覆盖分析")
    print("=" * 60)

    print(f"有效样本数：{total}")
    print(f"无效 JSON 行数：{result['invalid_json_lines']}")
    print(f"缺少 SQL 样本数：{result['missing_sql_samples']}")

    print("\n一、SQL 特征覆盖")
    for feature, count in result["feature_counts"].most_common():
        ratio = count / total * 100 if total else 0
        print(f"{feature:<15} {count:>8}  {ratio:>6.2f}%")

    print("\n二、聚合函数分布")
    for func, count in result["aggregate_counts"].most_common():
        print(f"{func:<15} {count:>8}")

    print("\n三、WHERE 条件数量分布")
    for count, num in sorted(
            result["where_condition_distribution"].items()
    ):
        print(f"{count} 个条件：{num} 条")

    print("\n四、SQL 结构组合 Top 20")
    for combination, count in result[
        "feature_combinations"
    ].most_common(20):
        name = " + ".join(combination) if combination else "无特征"
        print(f"{name:<50} {count:>8}")

    print("\n五、重复情况")
    print(f"归一化后不同 SQL 模板数：{result['unique_sql_patterns']}")
    print(f"重复冗余样本数：{result['duplicate_excess']}")
    print(f"重复率：{result['duplicate_rate']:.2%}")

    print("\n六、表覆盖情况")
    print(f"涉及表数量：{result['table_count']}")
    print("样本量最多的 20 张表：")
    for table_id, count in result["top_tables"]:
        print(f"{table_id:<30} {count:>8}")


if __name__ == "__main__":
    jsonl_path = "question_sql.jsonl"

    result = analyze_sql_dataset(jsonl_path)
    print_analysis_report(result)
