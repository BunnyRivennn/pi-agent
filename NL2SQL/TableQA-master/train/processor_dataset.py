import json
import sqlite3
from pathlib import Path
from decimal import Decimal
from datetime import date, datetime

# =========================
# 配置
# =========================

INPUT_FILE = "question_sql.jsonl"
OUTPUT_FILE = "question_sql_with_result.jsonl"

# 改成你的 .db 文件路径
DB_FILE = "train.db"

# 每条 SQL 最多保存多少行结果
MAX_RESULT_ROWS = 1000


def json_default(value):
    """将数据库结果转换成 JSON 可保存的类型。"""
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def execute_sql(conn, sql):
    """执行一条 SQL，返回结果和状态。"""

    if not isinstance(sql, str) or not sql.strip():
        return {
            "status": "error",
            "columns": [],
            "result": [],
            "row_count": 0,
            "truncated": False,
            "error": "SQL 为空"
        }

    try:
        cursor = conn.execute(sql)

        # SELECT 查询应当返回列信息
        if cursor.description is None:
            return {
                "status": "error",
                "columns": [],
                "result": [],
                "row_count": 0,
                "truncated": False,
                "error": "SQL 未返回结果集"
            }

        columns = [item[0] for item in cursor.description]

        # 多读取一行，判断结果是否被截断
        rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

        truncated = len(rows) > MAX_RESULT_ROWS
        rows = rows[:MAX_RESULT_ROWS]

        result = [
            [json_default(value) if value is not None else None
             for value in row]
            for row in rows
        ]

        return {
            "status": "success",
            "columns": columns,
            "result": result,
            "row_count": len(result),
            "truncated": truncated,
            "error": None
        }

    except Exception as e:
        return {
            "status": "error",
            "columns": [],
            "result": [],
            "row_count": 0,
            "truncated": False,
            "error": str(e)
        }


def main():
    input_path = Path(INPUT_FILE)
    db_path = Path(DB_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    if not db_path.exists():
        raise FileNotFoundError(f"找不到数据库文件：{db_path}")

    records = []

    # 读取 JSONL：一行一条 JSON
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

    print(f"读取到 {len(records)} 条数据")

    # SQLite 只读连接，避免误修改数据库
    db_uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)

    success_count = 0
    error_count = 0

    try:
        for index, record in enumerate(records, start=1):
            sql = record.get("sql")

            # 仅执行查询语句，避免 UPDATE/DELETE 等操作
            if not isinstance(sql, str) or not sql.lstrip().upper().startswith(
                    ("SELECT", "WITH")
            ):
                execution = {
                    "status": "blocked",
                    "columns": [],
                    "result": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": "只允许 SELECT 或 WITH 查询"
                }
            else:
                execution = execute_sql(conn, sql)

            record["execution"] = execution

            if execution["status"] == "success":
                success_count += 1
            else:
                error_count += 1

            if index % 100 == 0 or index == len(records):
                print(
                    f"进度：{index}/{len(records)}，"
                    f"成功：{success_count}，"
                    f"失败/拦截：{error_count}"
                )

    finally:
        conn.close()

    # 保存为 JSONL，保持一行一条记录
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n处理完成")
    print(f"成功：{success_count}")
    print(f"失败/拦截：{error_count}")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
