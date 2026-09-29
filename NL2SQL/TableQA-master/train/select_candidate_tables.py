import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_value(obj: dict, *keys, default=None):
    """按候选字段名依次取值。"""
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def normalize_tables(data: Any) -> list[dict]:
    """
    兼容常见格式：
    1. {"tables": [...]}
    2. {"tables": {"table_id": {...}}}
    3. [...]
    """
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("inspection JSON 顶层必须是 list 或 dict")

    tables = data.get("tables", data)

    if isinstance(tables, list):
        return tables

    if isinstance(tables, dict):
        result = []
        for table_id, info in tables.items():
            if not isinstance(info, dict):
                continue

            item = dict(info)
            item.setdefault("table_id", table_id)
            result.append(item)

        return result

    raise ValueError("无法识别 inspection JSON 中的 tables 结构")


def get_columns(table: dict) -> list[dict]:
    """
    优先使用 column_stats，因为它通常包含字段统计信息。
    如果没有，再尝试使用 columns。
    """
    stats = table.get("column_stats")
    if isinstance(stats, list):
        return stats

    columns = table.get("columns")
    if isinstance(columns, list):
        return columns

    return []


def normalize_type(value: Any) -> str:
    return str(value or "").strip().upper()


def is_numeric_type(value: Any) -> bool:
    value = normalize_type(value)
    return any(
        token in value
        for token in (
            "INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC",
            "DECIMAL", "NUMBER"
        )
    )


def is_text_type(value: Any) -> bool:
    value = normalize_type(value)
    return any(
        token in value
        for token in ("TEXT", "CHAR", "CLOB", "STRING", "VARCHAR")
    )


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_distinct_count(column: dict) -> int:
    value = get_value(
        column,
        "distinct_count",
        "distinct_values",
        default=0
    )

    if isinstance(value, list):
        return len(value)

    return to_int(value)


def get_non_null_count(column: dict) -> int:
    return to_int(
        get_value(
            column,
            "non_null_count",
            "non_null",
            default=0
        )
    )


def get_total_count(column: dict, table_row_count: int) -> int:
    return to_int(
        get_value(
            column,
            "total_count",
            "total",
            default=table_row_count
        ),
        default=table_row_count
    )


def classify_column(column: dict) -> str:
    """
    结合字段类型和字段统计信息进行粗略分类。
    返回 numeric / category / text / other。
    """
    declared_type = get_value(
        column,
        "declared_type",
        "type",
        "data_type",
        default=""
    )

    if is_numeric_type(declared_type):
        return "numeric"

    distinct_count = get_distinct_count(column)

    if is_text_type(declared_type):
        # 文本字段去重值较少时，更适合做类别筛选或分组。
        if 2 <= distinct_count <= 100:
            return "category"
        return "text"

    return "other"


def evaluate_table(
        table: dict,
        min_rows: int = 20,
        min_non_null_rate: float = 0.5,
        min_distinct_categories: int = 2,
        max_category_distinct: int = 100
) -> dict | None:
    table_id = get_value(
        table,
        "table_id",
        "id",
        default=""
    )

    table_name = get_value(
        table,
        "table_name",
        "name",
        default=f"Table_{table_id}"
    )

    row_count = to_int(
        get_value(
            table,
            "row_count",
            "rows",
            "record_count",
            default=0
        )
    )

    if row_count < min_rows:
        return None

    columns = get_columns(table)
    if not columns:
        return None

    numeric_columns = []
    category_columns = []
    text_columns = []
    usable_columns = []

    for column in columns:
        column_name = get_value(
            column,
            "column_name",
            "name",
            "column",
            default=""
        )

        declared_type = get_value(
            column,
            "declared_type",
            "type",
            "data_type",
            default=""
        )

        if not column_name:
            continue

        total_count = get_total_count(column, row_count)
        non_null_count = get_non_null_count(column)

        # 如果统计中没有 non_null_count，避免错误地把字段全部过滤。
        if total_count <= 0:
            non_null_rate = 0.0
        elif "non_null_count" not in column and "non_null" not in column:
            non_null_rate = 1.0
        else:
            non_null_rate = non_null_count / total_count

        if non_null_rate < min_non_null_rate:
            continue

        distinct_count = get_distinct_count(column)
        col_type = classify_column(column)

        column_info = {
            "column_name": column_name,
            "declared_type": declared_type,
            "distinct_count": distinct_count,
            "non_null_rate": round(non_null_rate, 4),
            "sample_values": column.get("sample_values", []),
            "possible_use": column.get("possible_use", [])
        }

        usable_columns.append(column_info)

        if col_type == "numeric":
            numeric_columns.append(column_info)

        elif col_type == "category":
            if min_distinct_categories <= distinct_count <= max_category_distinct:
                category_columns.append(column_info)

        elif col_type == "text":
            text_columns.append(column_info)

    # 至少需要两个可用字段，且要有数值字段或类别字段。
    if len(usable_columns) < 2:
        return None

    if not numeric_columns and not category_columns:
        return None

    recommended_tasks = []

    if category_columns:
        recommended_tasks.append("条件筛选")

    if numeric_columns:
        recommended_tasks.append("数值比较")
        recommended_tasks.append("聚合查询")

    if category_columns and numeric_columns:
        recommended_tasks.append("分组统计")

    if len(category_columns) >= 2:
        recommended_tasks.append("多条件筛选")

    # 简单评分，仅用于候选表排序，不代表 SQL 难度。
    score = 0
    score += min(row_count / 100, 20)
    score += min(len(numeric_columns) * 3, 15)
    score += min(len(category_columns) * 4, 20)
    score += min(len(text_columns) * 2, 10)

    reasons = [
        f"数据行数为 {row_count}",
        f"可用字段数为 {len(usable_columns)}",
        f"数值字段数为 {len(numeric_columns)}",
        f"类别字段数为 {len(category_columns)}"
    ]

    return {
        "table_id": table_id,
        "table_name": table_name,
        "row_count": row_count,
        "column_count": len(columns),
        "usable_column_count": len(usable_columns),
        "numeric_columns": numeric_columns,
        "category_columns": category_columns,
        "text_columns": text_columns,
        "recommended_tasks": recommended_tasks,
        "score": round(score, 2),
        "reasons": reasons
    }


def main():
    parser = argparse.ArgumentParser(
        description="从数据库检查报告中筛选 NL2SQL 合成样本候选表"
    )

    parser.add_argument(
        "--input",
        default="database_inspection.json",
        help="数据库检查报告路径"
    )

    parser.add_argument(
        "--output",
        default="candidate_tables.json",
        help="候选表输出路径"
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=20,
        help="候选表最少数据行数，默认 20"
    )

    parser.add_argument(
        "--min-non-null-rate",
        type=float,
        default=0.5,
        help="字段最低非空率，默认 0.5"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="最多保留多少张候选表，默认 1000"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_json(input_path)
    tables = normalize_tables(data)

    candidates = []

    for table in tables:
        if not isinstance(table, dict):
            continue

        candidate = evaluate_table(
            table,
            min_rows=args.min_rows,
            min_non_null_rate=args.min_non_null_rate
        )

        if candidate:
            candidates.append(candidate)

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if args.top_k > 0:
        candidates = candidates[:args.top_k]

    result = {
        "total_tables_scanned": len(tables),
        "candidate_table_count": len(candidates),
        "filter_config": {
            "min_rows": args.min_rows,
            "min_non_null_rate": args.min_non_null_rate,
            "top_k": args.top_k
        },
        "tables": candidates
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"扫描表数量：{len(tables)}")
    print(f"候选表数量：{len(candidates)}")
    print(f"结果已保存：{output_path.resolve()}")

    print("\n候选表推荐任务统计：")
    task_counts = {}

    for table in candidates:
        for task in table["recommended_tasks"]:
            task_counts[task] = task_counts.get(task, 0) + 1

    for task, count in sorted(
            task_counts.items(),
            key=lambda x: x[1],
            reverse=True
    ):
        print(f"{task}: {count}")


if __name__ == "__main__":
    main()
