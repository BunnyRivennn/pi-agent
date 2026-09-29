import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

# =========================
# 配置
# =========================

INPUT_FILE = "question_sql_labeled.jsonl"

REPORT_FILE = "sql_structure_report.json"
SAMPLES_FILE = "sql_structure_samples.jsonl"

# 每种当前难度随机抽取多少条
SAMPLES_PER_DIFFICULTY = 30

RANDOM_SEED = 42


# =========================
# SQL 处理
# =========================

def normalize_sql(sql):
    """去除注释、统一空白和大小写。"""
    if not isinstance(sql, str):
        return ""

    sql = re.sub(r"--.*?$", " ", sql, flags=re.M)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"\s+", " ", sql)

    return sql.strip().lower()


def count_pattern(pattern, sql):
    return len(re.findall(pattern, sql, flags=re.I))


def extract_where_clause(sql):
    """
    粗略提取 WHERE 内容。
    这是规则统计，不是完整 SQL 语法解析。
    """
    match = re.search(
        r"\bwhere\b(.*?)(\bgroup\s+by\b|\border\s+by\b|"
        r"\bhaving\b|\blimit\b|\bunion\b|$)",
        sql,
        flags=re.I
    )

    return match.group(1).strip() if match else ""


def count_conditions(sql):
    where_part = extract_where_clause(sql)

    if not where_part:
        return 0

    # 统计 AND / OR，条件数近似为连接符数量 + 1
    connectors = count_pattern(r"\b(and|or)\b", where_part)

    return connectors + 1


def count_selected_columns(sql):
    """
    粗略统计 SELECT 字段数。
    对 SELECT *、复杂表达式等情况只做近似判断。
    """
    match = re.search(
        r"\bselect\b(.*?)\bfrom\b",
        sql,
        flags=re.I
    )

    if not match:
        return 0

    select_part = match.group(1).strip()

    if select_part == "*":
        return 1

    # 简单按逗号拆分
    return len([x for x in select_part.split(",") if x.strip()])


def count_tables(sql):
    """
    统计 FROM/JOIN 后的表引用数量。
    主要用于常见的简单 SQL。
    """
    from_count = count_pattern(r"\bfrom\b", sql)
    join_count = count_pattern(r"\bjoin\b", sql)

    return from_count + join_count


# =========================
# 单条 SQL 特征分析
# =========================

def analyze_sql(sql):
    normalized = normalize_sql(sql)

    features = {
        "has_where": bool(re.search(r"\bwhere\b", normalized)),
        "has_and": bool(re.search(r"\band\b", normalized)),
        "has_or": bool(re.search(r"\bor\b", normalized)),
        "has_group_by": bool(re.search(r"\bgroup\s+by\b", normalized)),
        "has_having": bool(re.search(r"\bhaving\b", normalized)),
        "has_order_by": bool(re.search(r"\border\s+by\b", normalized)),
        "has_limit": bool(re.search(r"\blimit\b", normalized)),
        "has_join": bool(re.search(r"\bjoin\b", normalized)),
        "has_subquery": bool(
            re.search(r"\(\s*select\b", normalized)
        ),
        "has_union": bool(re.search(r"\bunion\b", normalized)),
        "has_with": bool(re.search(r"\bwith\b", normalized)),
        "has_distinct": bool(re.search(r"\bdistinct\b", normalized)),
        "has_aggregate": bool(
            re.search(
                r"\b(count|sum|avg|max|min)\s*\(",
                normalized
            )
        ),
        "has_case": bool(re.search(r"\bcase\b", normalized)),
        "has_in": bool(re.search(r"\bin\s*\(", normalized)),
        "has_like": bool(re.search(r"\blike\b", normalized)),
        "has_between": bool(re.search(r"\bbetween\b", normalized)),
        "has_not": bool(re.search(r"\bnot\b", normalized)),
        "condition_count": count_conditions(normalized),
        "select_column_count": count_selected_columns(normalized),
        "table_reference_count": count_tables(normalized),
        "sql_length": len(normalized),
    }

    return features


# =========================
# 数据读取
# =========================

def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"第 {line_no} 行 JSON 格式错误：{e}"
                )

    return records


# =========================
# 主流程
# =========================

def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    records = load_jsonl(INPUT_FILE)

    print(f"读取到 {len(records)} 条数据")

    feature_counts = Counter()
    difficulty_counts = Counter()
    feature_by_difficulty = defaultdict(Counter)

    samples_by_difficulty = defaultdict(list)

    for record in records:
        sql = record.get("sql", "")
        difficulty = record.get("difficulty", "unknown")

        features = analyze_sql(sql)

        difficulty_counts[difficulty] += 1

        for key, value in features.items():
            if isinstance(value, bool):
                if value:
                    feature_counts[key] += 1
                    feature_by_difficulty[difficulty][key] += 1

        # 数值特征单独统计
        feature_counts["condition_count_total"] += features[
            "condition_count"
        ]
        feature_counts["select_column_count_total"] += features[
            "select_column_count"
        ]
        feature_counts["table_reference_count_total"] += features[
            "table_reference_count"
        ]
        feature_counts["sql_length_total"] += features["sql_length"]

        samples_by_difficulty[difficulty].append({
            "question": record.get("question"),
            "sql": sql,
            "difficulty": difficulty,
            "difficulty_reason": record.get("difficulty_reason"),
            "execution_status": record.get("execution", {}).get("status"),
            "features": features,
        })

    # =========================
    # 随机抽样
    # =========================

    random.seed(RANDOM_SEED)

    sampled_records = []

    for difficulty, samples in samples_by_difficulty.items():
        sample_size = min(
            SAMPLES_PER_DIFFICULTY,
            len(samples)
        )

        sampled_records.extend(
            random.sample(samples, sample_size)
        )

    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        for record in sampled_records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # =========================
    # 汇总报告
    # =========================

    total = len(records)

    report = {
        "total_samples": total,
        "difficulty_counts": dict(difficulty_counts),
        "feature_counts": dict(feature_counts),
        "feature_by_difficulty": {
            difficulty: dict(counter)
            for difficulty, counter in feature_by_difficulty.items()
        },
        "sample_counts": {
            difficulty: len(samples)
            for difficulty, samples in samples_by_difficulty.items()
        },
        "config": {
            "samples_per_difficulty": SAMPLES_PER_DIFFICULTY,
            "random_seed": RANDOM_SEED,
        }
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # =========================
    # 控制台输出
    # =========================

    print("\n========== 难度分布 ==========")

    for difficulty, count in difficulty_counts.most_common():
        ratio = count / total * 100 if total else 0
        print(f"{difficulty:10s} {count:6d} ({ratio:.2f}%)")

    print("\n========== SQL 结构统计 ==========")

    boolean_features = [
        "has_where",
        "has_and",
        "has_or",
        "has_group_by",
        "has_having",
        "has_order_by",
        "has_limit",
        "has_join",
        "has_subquery",
        "has_union",
        "has_with",
        "has_distinct",
        "has_aggregate",
        "has_case",
        "has_in",
        "has_like",
        "has_between",
        "has_not",
    ]

    for feature in boolean_features:
        count = feature_counts.get(feature, 0)
        ratio = count / total * 100 if total else 0
        print(f"{feature:20s} {count:6d} ({ratio:.2f}%)")

    print("\n========== 平均结构特征 ==========")

    if total:
        print(
            "平均 WHERE 条件数：",
            round(
                feature_counts["condition_count_total"] / total,
                2
            )
        )
        print(
            "平均 SELECT 字段数：",
            round(
                feature_counts["select_column_count_total"] / total,
                2
            )
        )
        print(
            "平均表引用数：",
            round(
                feature_counts["table_reference_count_total"] / total,
                2
            )
        )
        print(
            "平均 SQL 长度：",
            round(
                feature_counts["sql_length_total"] / total,
                2
            )
        )

    print("\n========== 按难度统计 SQL 特征 ==========")

    for difficulty in ("easy", "medium", "hard", "invalid", "unknown"):
        if difficulty not in samples_by_difficulty:
            continue

        print(f"\n[{difficulty}]")

        for feature in boolean_features:
            count = feature_by_difficulty[difficulty].get(feature, 0)
            print(f"{feature:20s} {count:6d}")

    print("\n分析完成")
    print(f"统计报告：{REPORT_FILE}")
    print(f"抽样文件：{SAMPLES_FILE}")


if __name__ == "__main__":
    main()
