import json
import sqlite3
from pathlib import Path
from collections import Counter

# =========================
# 配置
# =========================

DB_FILE = r"train.db"

OUTPUT_FILE = "database_inspection.json"

# 每个字段最多采样多少个不同值
MAX_SAMPLE_VALUES = 10

# 每张表最多扫描多少行来统计字段不同值
# 设置为 None 表示扫描整张表
MAX_SCAN_ROWS = 10000


def quote_identifier(name):
    """安全引用 SQLite 表名或字段名。"""
    return '"' + name.replace('"', '""') + '"'


def get_tables(conn):
    """获取数据库中的所有普通表。"""
    rows = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """).fetchall()

    return [row[0] for row in rows]


def get_table_columns(conn, table_name):
    """获取表结构。"""
    sql = f"PRAGMA table_info({quote_identifier(table_name)})"
    rows = conn.execute(sql).fetchall()

    columns = []

    for row in rows:
        columns.append({
            "cid": row[0],
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default_value": row[4],
            "primary_key": bool(row[5]),
        })

    return columns


def get_row_count(conn, table_name):
    """获取表的总行数。"""
    sql = f"SELECT COUNT(*) FROM {quote_identifier(table_name)}"
    return conn.execute(sql).fetchone()[0]


def get_foreign_keys(conn, table_name):
    """读取 SQLite 显式声明的外键。"""
    sql = f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
    rows = conn.execute(sql).fetchall()

    foreign_keys = []

    for row in rows:
        foreign_keys.append({
            "referenced_table": row[2],
            "from_column": row[3],
            "to_column": row[4],
        })

    return foreign_keys


def get_column_stats(conn, table_name, column):
    """统计单个字段的数据分布。"""
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column["name"])

    total_count = get_row_count(conn, table_name)

    non_null_sql = f"""
        SELECT COUNT({column_sql})
        FROM {table_sql}
    """
    non_null_count = conn.execute(non_null_sql).fetchone()[0]

    distinct_sql = f"""
        SELECT COUNT(DISTINCT {column_sql})
        FROM {table_sql}
    """
    distinct_count = conn.execute(distinct_sql).fetchone()[0]

    sample_sql = f"""
        SELECT DISTINCT {column_sql}
        FROM {table_sql}
        WHERE {column_sql} IS NOT NULL
        LIMIT ?
    """
    sample_values = conn.execute(
        sample_sql,
        (MAX_SAMPLE_VALUES,)
    ).fetchall()

    sample_values = [row[0] for row in sample_values]

    stats = {
        "column_name": column["name"],
        "declared_type": column["type"],
        "total_count": total_count,
        "non_null_count": non_null_count,
        "null_count": total_count - non_null_count,
        "distinct_count": distinct_count,
        "sample_values": sample_values,
    }

    declared_type = (column["type"] or "").upper()

    # 数值字段统计
    if any(
            keyword in declared_type
            for keyword in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DECIMAL")
    ):
        numeric_sql = f"""
            SELECT
                MIN({column_sql}),
                MAX({column_sql}),
                AVG({column_sql})
            FROM {table_sql}
        """

        row = conn.execute(numeric_sql).fetchone()

        stats["numeric_stats"] = {
            "min": row[0],
            "max": row[1],
            "avg": row[2],
        }

    # 简单字段用途建议
    if distinct_count <= 30:
        stats["possible_use"] = [
            "filter",
            "group_by_candidate"
        ]
    elif distinct_count / max(non_null_count, 1) > 0.8:
        stats["possible_use"] = [
            "entity_or_id_candidate"
        ]
    else:
        stats["possible_use"] = [
            "filter_candidate"
        ]

    if any(
            keyword in declared_type
            for keyword in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DECIMAL")
    ):
        stats["possible_use"].append("aggregation_candidate")

    return stats


def inspect_table(conn, table_name):
    """分析一张表。"""
    columns = get_table_columns(conn, table_name)
    row_count = get_row_count(conn, table_name)

    column_stats = []

    for column in columns:
        stats = get_column_stats(conn, table_name, column)
        column_stats.append(stats)

    return {
        "table_name": table_name,
        "row_count": row_count,
        "columns": columns,
        "column_stats": column_stats,
        "foreign_keys": get_foreign_keys(conn, table_name),
    }


def find_possible_join_columns(tables_data):
    """
    根据同名字段，寻找可能的关联字段。
    这里只提供候选，不代表真实外键关系。
    """
    column_map = {}

    for table in tables_data:
        table_name = table["table_name"]

        for column in table["columns"]:
            column_name = column["name"]

            column_map.setdefault(column_name, []).append(table_name)

    candidates = []

    for column_name, table_names in column_map.items():
        unique_tables = sorted(set(table_names))

        if len(unique_tables) >= 2:
            candidates.append({
                "column_name": column_name,
                "tables": unique_tables,
                "note": "同名字段候选，需人工确认是否可关联",
            })

    return candidates


def main():
    db_path = Path(DB_FILE)

    if not db_path.exists():
        raise FileNotFoundError(
            f"找不到数据库文件：{DB_FILE}"
        )

    # SQLite 只读连接
    db_uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)

    try:
        tables = get_tables(conn)

        print(f"发现 {len(tables)} 张表")

        tables_data = []

        for index, table_name in enumerate(tables, start=1):
            print(f"正在分析：{table_name} ({index}/{len(tables)})")

            table_info = inspect_table(conn, table_name)
            tables_data.append(table_info)

        report = {
            "database_file": str(db_path.resolve()),
            "table_count": len(tables_data),
            "tables": tables_data,
            "possible_join_columns": find_possible_join_columns(
                tables_data
            ),
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                ensure_ascii=False,
                indent=2
            )

        print("\n========== 数据库概览 ==========")

        for table in tables_data:
            print(
                f"{table['table_name']}: "
                f"{table['row_count']} 行，"
                f"{len(table['columns'])} 个字段"
            )

        print("\n========== 可能的关联字段 ==========")

        for item in report["possible_join_columns"]:
            print(
                f"{item['column_name']}: "
                f"{', '.join(item['tables'])}"
            )

        print("\n分析完成")
        print(f"报告文件：{OUTPUT_FILE}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
