import json
from pathlib import Path

TRAIN_FILE = Path("train.json")
TABLES_FILE = Path("train.tables.json")
OUTPUT_FILE = Path("question_sql.jsonl")

# 数据集标注中的聚合操作映射
AGG_MAP = {
    0: None,
    1: "MAX",
    2: "MIN",
    3: "COUNT",
    4: "SUM",
    5: "SUM",
}

# 数据集标注中的条件操作符映射
OP_MAP = {
    0: "!=",
    1: ">",
    2: "=",
    3: "<",
    4: ">=",
    5: "<=",
    6: "!=",
    7: "LIKE",
    8: "IN",
    9: "NOT IN",
    10: "!=",
}


def read_jsonl(path):
    """逐行读取 JSONL 文件。"""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path} 第 {line_no} 行不是合法 JSON：{e}"
                )

    return records


def quote_sql_string(value):
    """将值转换为 SQL 字符串字面量。"""
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        return str(value)

    # SQL 字符串中的单引号需要转义为两个单引号
    value = str(value).replace("'", "''")
    return f"'{value}'"


def get_table_id(table):
    """兼容常见的表 ID 字段名。"""
    return (
            table.get("id")
            or table.get("table_id")
            or table.get("tableId")
    )


def get_table_name(table, table_id):
    """优先使用 tables 文件里的 name，否则使用 Table_<id>。"""
    name = table.get("name")

    if name:
        return name

    return f"Table_{table_id}"


def column_name(index):
    """列索引从 0 开始，物理字段名从 col_1 开始。"""
    return f"col_{index + 1}"


def build_sql(record, table_info):
    question = record["question"]
    table_id = record["table_id"]
    sql_data = record["sql"]

    table_name = get_table_name(table_info, table_id)

    selected_columns = sql_data.get("sel", [])
    aggregations = sql_data.get("agg", [])

    if not selected_columns:
        raise ValueError(f"问题没有 sel 字段：{question}")

    select_parts = []

    for i, col_idx in enumerate(selected_columns):
        col = column_name(col_idx)

        agg_idx = aggregations[i] if i < len(aggregations) else 0
        agg = AGG_MAP.get(agg_idx)

        if agg:
            select_parts.append(f"{agg}({col})")
        else:
            select_parts.append(col)

    sql = (
        f"SELECT {', '.join(select_parts)}\n"
        f"FROM {table_name}"
    )

    conditions = sql_data.get("conds", [])
    condition_parts = []

    for condition in conditions:
        col_idx, op_idx, value = condition

        col = column_name(col_idx)
        op = OP_MAP.get(op_idx)

        if op is None:
            raise ValueError(
                f"未知操作符编号 {op_idx}，问题：{question}"
            )

        if op in ("IN", "NOT IN"):
            if isinstance(value, list):
                values = value
            else:
                values = [value]

            values_sql = ", ".join(
                quote_sql_string(v) for v in values
            )

            condition_parts.append(
                f"{col} {op} ({values_sql})"
            )
        else:
            condition_parts.append(
                f"{col} {op} {quote_sql_string(value)}"
            )

    if condition_parts:
        conn_op = sql_data.get("cond_conn_op", 0)

        if conn_op == 2:
            connector = "OR"
        else:
            connector = "AND"

        sql += "\nWHERE " + f" {connector} ".join(condition_parts)

    return sql + ";"


def main():
    train_records = read_jsonl(TRAIN_FILE)
    table_records = read_jsonl(TABLES_FILE)

    # 建立 table_id -> 表结构映射
    table_map = {}

    for table in table_records:
        table_id = get_table_id(table)

        if not table_id:
            continue

        table_map[table_id] = table

    output_count = 0
    skipped_count = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for record in train_records:
            table_id = record.get("table_id")

            table_info = table_map.get(table_id)

            if table_info is None:
                print(f"跳过：找不到表结构，table_id={table_id}")
                skipped_count += 1
                continue

            try:
                sql = build_sql(record, table_info)

                output_record = {
                    "question": record["question"],
                    "table_id": table_id,
                    "sql": sql,
                }

                out.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )

                output_count += 1

            except Exception as e:
                print(
                    f"转换失败：{record.get('question', '')}\n"
                    f"原因：{e}"
                )
                skipped_count += 1

    print(f"转换完成：{output_count} 条")
    print(f"跳过：{skipped_count} 条")
    print(f"输出文件：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
