import json
import re
from pathlib import Path

# =========================
# 配置
# =========================

INPUT_FILE = "question_sql_with_result.jsonl"
OUTPUT_FILE = "question_sql_labeled.jsonl"

DIFFICULTY_VERSION = "rule_v1"


def normalize_sql(sql: str) -> str:
    """统一 SQL 格式，便于规则判断。"""
    sql = sql or ""
    sql = re.sub(r"--.*?$", " ", sql, flags=re.M)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower()


def count_conditions(sql: str) -> int:
    """
    粗略统计 WHERE 中的条件数量。
    这里按 AND / OR 估算，不做完整 SQL 语法解析。
    """
    match = re.search(
        r"\bwhere\b(.*?)(\bgroup\s+by\b|\border\s+by\b|"
        r"\bhaving\b|\blimit\b|$)",
        sql,
        flags=re.I
    )

    if not match:
        return 0

    where_part = match.group(1)

    # 简单按 AND / OR 拆分
    parts = re.split(r"\s+\band\b\s+|\s+\bor\b\s+", where_part)
    return len([p for p in parts if p.strip()])


def classify_difficulty(record: dict):
    """
    返回 difficulty 和 difficulty_reason。

    标签：
    - invalid：SQL 执行失败或被拦截
    - easy：简单单表查询
    - medium：聚合、排序、多条件等
    - hard：JOIN、子查询、UNION、窗口函数等
    """

    execution = record.get("execution", {})

    # SQL 执行失败，不参与正常难度统计
    if execution.get("status") != "success":
        return "invalid", "SQL 执行失败或被拦截"

    sql = normalize_sql(record.get("sql", ""))

    if not sql:
        return "invalid", "SQL 为空"

    # ---------- Hard：复杂 SQL 结构 ----------

    hard_patterns = [
        (r"\bjoin\b", "包含 JOIN 多表关联"),
        (r"\bunion\b", "包含 UNION 合并查询"),
        (r"\bintersect\b", "包含 INTERSECT 集合操作"),
        (r"\bexcept\b", "包含 EXCEPT 集合操作"),
        (r"\bexists\s*\(", "包含 EXISTS 子查询"),
        (r"\bover\s*\(", "包含窗口函数"),
        (r"\bwith\b", "包含 WITH 公共表表达式"),
    ]

    for pattern, reason in hard_patterns:
        if re.search(pattern, sql):
            return "hard", reason

    # SELECT 中出现括号包裹的 SELECT，视为子查询
    if re.search(r"\(\s*select\b", sql):
        return "hard", "包含子查询"

    # ---------- Medium：聚合、分组、排序等 ----------

    medium_patterns = [
        (r"\bgroup\s+by\b", "包含 GROUP BY 分组"),
        (r"\bhaving\b", "包含 HAVING 分组筛选"),
        (r"\border\s+by\b", "包含 ORDER BY 排序"),
        (r"\blimit\b", "包含 LIMIT 限制结果数量"),
        (r"\bavg\s*\(", "包含 AVG 聚合"),
        (r"\bsum\s*\(", "包含 SUM 聚合"),
        (r"\bcount\s*\(", "包含 COUNT 聚合"),
        (r"\bmax\s*\(", "包含 MAX 聚合"),
        (r"\bmin\s*\(", "包含 MIN 聚合"),
        (r"\bdistinct\b", "包含 DISTINCT 去重"),
    ]

    for pattern, reason in medium_patterns:
        if re.search(pattern, sql):
            return "medium", reason

    # 多条件筛选
    condition_count = count_conditions(sql)

    if condition_count >= 3:
        return "medium", f"WHERE 中约有 {condition_count} 个筛选条件"

    # 多列查询
    select_match = re.search(
        r"\bselect\b(.*?)\bfrom\b",
        sql,
        flags=re.I
    )

    if select_match:
        select_part = select_match.group(1).strip()

        # 不把 COUNT(*) 等表达式中的逗号简单当作多列
        if "," in select_part:
            return "medium", "SELECT 查询多个字段"

    # ---------- Easy：默认简单查询 ----------

    if condition_count >= 1:
        return "easy", f"单表基础查询，约有 {condition_count} 个筛选条件"

    return "easy", "单表基础查询，无复杂 SQL 结构"


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    records = []

    # JSONL：每行一条记录
    with input_path.open("r", encoding="utf-8") as f:
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

    counts = {
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "invalid": 0,
    }

    for record in records:
        difficulty, reason = classify_difficulty(record)

        record["difficulty"] = difficulty
        record["difficulty_reason"] = reason
        record["difficulty_version"] = DIFFICULTY_VERSION

        counts[difficulty] = counts.get(difficulty, 0) + 1

    # 保存为 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    total = len(records)

    print(f"处理完成，共 {total} 条")
    print(f"Easy:    {counts['easy']}")
    print(f"Medium:  {counts['medium']}")
    print(f"Hard:    {counts['hard']}")
    print(f"Invalid: {counts['invalid']}")

    valid_count = total - counts["invalid"]

    if valid_count > 0:
        print("\n有效样本难度占比：")
        for label in ("easy", "medium", "hard"):
            ratio = counts[label] / valid_count * 100
            print(f"{label}: {ratio:.2f}%")

    print(f"\n输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()


"""
有效样本难度占比：
easy: 75.15%
medium: 24.85%
hard: 0.00%
"""