import json
import shutil
import sqlite3
import time
from pathlib import Path

# ============================================================
# 配置
# ============================================================

TRAIN_JSON = "train.json"
TABLES_JSON = "train.tables.json"
INPUT_DB = "train.db"

OUTPUT_DB = "train_renamed.db"

OUTPUT_SQL = "questions_sql.jsonl"
OUTPUT_SUCCESS = "sql_success.jsonl"
OUTPUT_ERROR = "sql_error.jsonl"

# ============================================================
# SQLite配置
# ============================================================

BATCH_SIZE = 200


# ============================================================
# JSONL读取
# ============================================================

def read_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))

            except json.JSONDecodeError as e:

                print(
                    f"[WARN] JSON解析失败: "
                    f"{path} 第 {line_no} 行"
                )

    return data


# ============================================================
# SQLite标识符
# ============================================================

def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


# ============================================================
# SQL值
# ============================================================

def quote_value(value):
    if value is None:
        return "NULL"

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):
        return str(value)

    return "'" + str(value).replace("'", "''") + "'"


# ============================================================
# 处理字段名
# ============================================================

def normalize_headers(headers):
    """
    将 header 转换成 SQLite 可以使用的唯一字段名。

    例如：

    [
        "排名",
        "周涨幅（%）",
        "周涨幅（%）",
        None,
        ""
    ]

    转换成：

    [
        "排名",
        "周涨幅（%）",
        "周涨幅（%）_2",
        "None",
        "unnamed_5"
    ]
    """

    result = []

    used = set()

    for index, header in enumerate(headers, 1):

        # ----------------------------------------------------
        # None
        # ----------------------------------------------------

        if header is None:
            name = "None"

        else:
            name = str(header).strip()

        # ----------------------------------------------------
        # 空字符串
        # ----------------------------------------------------

        if not name:
            name = f"unnamed_{index}"

        # ----------------------------------------------------
        # 重复字段
        # ----------------------------------------------------

        original_name = name

        counter = 2

        while name in used:
            name = f"{original_name}_{counter}"

            counter += 1

        used.add(name)

        result.append(name)

    return result


# ============================================================
# 加载表结构
# ============================================================

def load_table_schema(tables_json):
    result = {}

    for item in tables_json:

        table_id = item.get("id")

        table_name = item.get("name")

        if not table_id and table_name:

            if table_name.startswith("Table_"):
                table_id = table_name[6:]

        if not table_id:
            continue

        result[table_id] = {
            "name": table_name,
            "header": item.get("header", [])
        }

    return result


# ============================================================
# SQLite表是否存在
# ============================================================

def table_exists(conn, table_name):
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        """,
        (table_name,)
    ).fetchone()

    return row is not None


# ============================================================
# 获取表结构
# ============================================================

def get_table_info(conn, table_name):
    return conn.execute(
        f"PRAGMA table_info({quote_identifier(table_name)})"
    ).fetchall()


# ============================================================
# 重建一张表
# ============================================================

def rebuild_table(conn, table_name, headers):
    # --------------------------------------------------------
    # 获取原字段
    # --------------------------------------------------------

    table_info = get_table_info(
        conn,
        table_name
    )

    if not table_info:
        print(
            f"[WARN] 表不存在: {table_name}"
        )

        return False

    old_columns = [
        row[1]
        for row in table_info
    ]

    # --------------------------------------------------------
    # 数量检查
    # --------------------------------------------------------

    if len(old_columns) != len(headers):
        print(
            f"[ERROR] 字段数量不一致: {table_name}"
        )

        print(
            f"       DB: {len(old_columns)}"
        )

        print(
            f"       JSON: {len(headers)}"
        )

        return False

    # --------------------------------------------------------
    # 处理重复字段
    # --------------------------------------------------------

    new_columns = normalize_headers(
        headers
    )

    # --------------------------------------------------------
    # 如果字段已经正确，跳过
    # --------------------------------------------------------

    if old_columns == new_columns:
        return True

    # --------------------------------------------------------
    # 临时表
    # --------------------------------------------------------

    temp_table = (
        f"__new_{table_name}"
    )

    # 如果之前残留，删除
    conn.execute(
        f"DROP TABLE IF EXISTS "
        f"{quote_identifier(temp_table)}"
    )

    # --------------------------------------------------------
    # 获取原表CREATE SQL
    # --------------------------------------------------------

    create_row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type='table'
          AND name=?
        """,
        (table_name,)
    ).fetchone()

    if not create_row:
        print(
            f"[ERROR] 找不到CREATE TABLE: "
            f"{table_name}"
        )

        return False

    # --------------------------------------------------------
    # 创建新表
    #
    # 尽量保持原字段类型
    # --------------------------------------------------------

    column_defs = []

    for i, row in enumerate(table_info):

        column_type = row[2]

        new_name = new_columns[i]

        if column_type:
            definition = (
                f"{quote_identifier(new_name)} "
                f"{column_type}"
            )

        else:
            definition = quote_identifier(
                new_name
            )

        column_defs.append(
            definition
        )

    create_sql = (
        f"CREATE TABLE "
        f"{quote_identifier(temp_table)} "
        f"({', '.join(column_defs)})"
    )

    conn.execute(create_sql)

    # --------------------------------------------------------
    # 复制数据
    # --------------------------------------------------------

    old_columns_sql = ", ".join(
        quote_identifier(c)
        for c in old_columns
    )

    new_columns_sql = ", ".join(
        quote_identifier(c)
        for c in new_columns
    )

    insert_sql = f"""
        INSERT INTO {quote_identifier(temp_table)}
        ({new_columns_sql})
        SELECT {old_columns_sql}
        FROM {quote_identifier(table_name)}
    """

    conn.execute(insert_sql)

    # --------------------------------------------------------
    # 删除旧表
    # --------------------------------------------------------

    conn.execute(
        f"DROP TABLE "
        f"{quote_identifier(table_name)}"
    )

    # --------------------------------------------------------
    # 新表改回原名
    # --------------------------------------------------------

    conn.execute(
        f"ALTER TABLE "
        f"{quote_identifier(temp_table)} "
        f"RENAME TO "
        f"{quote_identifier(table_name)}"
    )

    return True


# ============================================================
# 重建所有表
# ============================================================

def rebuild_database(conn, table_schema):
    print()
    print("=" * 70)
    print("开始高速重建数据库字段")
    print("=" * 70)

    total = len(table_schema)

    success = 0
    failed = 0

    start_time = time.time()

    # --------------------------------------------------------
    # 整体事务
    # --------------------------------------------------------

    conn.execute("BEGIN")

    try:

        for index, (table_id, info) in enumerate(
                table_schema.items(),
                1
        ):

            table_name = info["name"]

            headers = info["header"]

            if not table_name:
                failed += 1

                continue

            try:

                ok = rebuild_table(
                    conn,
                    table_name,
                    headers
                )

                if ok:
                    success += 1
                else:
                    failed += 1

            except Exception as e:

                failed += 1

                print()
                print(
                    f"[ERROR] {table_name}"
                )

                print(e)

            # ------------------------------------------------
            # 进度
            # ------------------------------------------------

            if (
                    index % 100 == 0
                    or index == total
            ):
                elapsed = (
                        time.time()
                        - start_time
                )

                speed = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[进度] "
                    f"{index}/{total} "
                    f"({index / total * 100:.1f}%) "
                    f"| 成功 {success} "
                    f"| 失败 {failed} "
                    f"| {speed:.1f} 表/秒"
                )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    elapsed = time.time() - start_time

    print()
    print("=" * 70)

    print(
        f"数据库重建完成"
    )

    print(
        f"总表数: {total}"
    )

    print(
        f"成功: {success}"
    )

    print(
        f"失败: {failed}"
    )

    print(
        f"耗时: {elapsed:.2f} 秒"
    )

    print("=" * 70)


# ============================================================
# Aggregation
# ============================================================

AGG_MAP = {
    0: None,
    1: "MAX",
    2: "MIN",
    3: "COUNT",
    4: "AVG",
    5: "SUM",
}

# ============================================================
# Operator
# ============================================================

OPERATOR_MAP = {
    0: "=",
    1: ">",
    2: "=",
    3: "<",
    4: ">=",
    5: "<=",
    6: "!=",
}

# ============================================================
# 条件连接
# ============================================================

CONN_MAP = {
    1: "AND",
    2: "OR",
}


# ============================================================
# 构造WHERE
# ============================================================

def build_conditions(sql_info, headers):
    conds = sql_info.get(
        "conds",
        []
    )

    if not conds:
        return ""

    conn_op = sql_info.get(
        "cond_conn_op",
        1
    )

    connector = CONN_MAP.get(
        conn_op,
        "AND"
    )

    # --------------------------------------------------------
    # 先按照字段分组
    # --------------------------------------------------------

    grouped_equal = {}

    normal_conditions = []

    for condition in conds:

        if len(condition) < 3:
            continue

        col_index = condition[0]

        operator = condition[1]

        value = condition[2]

        if (
                col_index < 0
                or col_index >= len(headers)
        ):
            continue

        column_name = headers[
            col_index
        ]

        # operator=2 → =
        if operator == 2:

            grouped_equal.setdefault(
                col_index,
                []
            ).append(value)

        else:

            normal_conditions.append(
                (
                    column_name,
                    operator,
                    value
                )
            )

    conditions = []

    # --------------------------------------------------------
    # 相同字段多个 =
    #
    # A=x OR A=y
    #
    # 转换为：
    #
    # A IN (x,y)
    # --------------------------------------------------------

    for col_index, values in grouped_equal.items():

        column_name = headers[
            col_index
        ]

        column_sql = quote_identifier(
            column_name
        )

        if len(values) == 1:

            conditions.append(
                f"{column_sql} = "
                f"{quote_value(values[0])}"
            )

        else:

            values_sql = ", ".join(
                quote_value(v)
                for v in values
            )

            conditions.append(
                f"{column_sql} IN "
                f"({values_sql})"
            )

    # --------------------------------------------------------
    # 普通条件
    # --------------------------------------------------------

    for column_name, operator, value in normal_conditions:

        op = OPERATOR_MAP.get(
            operator
        )

        if op is None:
            print(
                f"[WARN] 未知operator={operator}"
            )

            continue

        conditions.append(
            f"{quote_identifier(column_name)} "
            f"{op} "
            f"{quote_value(value)}"
        )

    return (
        f" {connector} "
    ).join(conditions)


# ============================================================
# 生成SQL
# ============================================================

def generate_sql(
        item,
        table_schema
):
    table_id = item.get(
        "table_id"
    )

    if table_id not in table_schema:
        raise ValueError(
            f"找不到table_id: {table_id}"
        )

    table_info = table_schema[
        table_id
    ]

    table_name = table_info[
        "name"
    ]

    # 注意：
    # SQL必须使用数据库真正的字段名。
    # 如果header有重复，我们必须使用normalize后的名字。
    headers = normalize_headers(
        table_info["header"]
    )

    sql_info = item.get(
        "sql",
        {}
    )

    sel = sql_info.get(
        "sel",
        []
    )

    agg = sql_info.get(
        "agg",
        []
    )

    if not sel:
        raise ValueError(
            "sql.sel为空"
        )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    select_parts = []

    for i, col_index in enumerate(sel):

        if (
                col_index < 0
                or col_index >= len(headers)
        ):
            raise ValueError(
                f"sel字段越界: {col_index}"
            )

        column_name = headers[
            col_index
        ]

        column_sql = quote_identifier(
            column_name
        )

        agg_value = (
            agg[i]
            if i < len(agg)
            else 0
        )

        aggregation = AGG_MAP.get(
            agg_value
        )

        if aggregation:
            column_sql = (
                f"{aggregation}"
                f"({column_sql})"
            )

        select_parts.append(
            column_sql
        )

    select_sql = ", ".join(
        select_parts
    )

    # --------------------------------------------------------
    # FROM
    # --------------------------------------------------------

    sql = (
        f"SELECT {select_sql}\n"
        f"FROM "
        f"{quote_identifier(table_name)}"
    )

    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    where_sql = build_conditions(
        sql_info,
        headers
    )

    if where_sql:
        sql += (
            f"\nWHERE {where_sql}"
        )

    sql += ";"

    return sql


# ============================================================
# 执行SQL
# ============================================================

def execute_sql(
        conn,
        sql
):
    try:

        cursor = conn.execute(
            sql
        )

        rows = cursor.fetchall()

        columns = []

        if cursor.description:
            columns = [
                x[0]
                for x in cursor.description
            ]

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "error": None
        }

    except Exception as e:

        return {
            "success": False,
            "columns": [],
            "rows": [],
            "error": str(e)
        }


# ============================================================
# JSON序列化
# ============================================================

def json_default(obj):
    if isinstance(
            obj,
            bytes
    ):
        return obj.decode(
            "utf-8",
            errors="replace"
        )

    return str(obj)


# ============================================================
# 主程序
# ============================================================

def main():
    total_start = time.time()

    print("=" * 70)
    print(
        "NL2SQL V2 "
        "高速数据库转换 + SQL验证"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 检查文件
    # --------------------------------------------------------

    for path in [
        TRAIN_JSON,
        TABLES_JSON,
        INPUT_DB
    ]:

        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到文件: {path}"
            )

    # --------------------------------------------------------
    # 读取表结构
    # --------------------------------------------------------

    print()
    print("[1] 读取 train.tables.json")

    tables_json = read_jsonl(
        TABLES_JSON
    )

    table_schema = load_table_schema(
        tables_json
    )

    print(
        f"    表结构: "
        f"{len(tables_json)}"
    )

    print(
        f"    有效table_id: "
        f"{len(table_schema)}"
    )

    # --------------------------------------------------------
    # 读取训练数据
    # --------------------------------------------------------

    print()
    print("[2] 读取 train.json")

    train_data = read_jsonl(
        TRAIN_JSON
    )

    print(
        f"    问题数量: "
        f"{len(train_data)}"
    )

    # --------------------------------------------------------
    # 复制数据库
    # --------------------------------------------------------

    print()
    print("[3] 复制数据库")

    if Path(OUTPUT_DB).exists():
        print(
            f"    删除旧文件: "
            f"{OUTPUT_DB}"
        )

        Path(OUTPUT_DB).unlink()

    shutil.copy2(
        INPUT_DB,
        OUTPUT_DB
    )

    print(
        f"    {INPUT_DB} -> "
        f"{OUTPUT_DB}"
    )

    # --------------------------------------------------------
    # 打开数据库
    # --------------------------------------------------------

    conn = sqlite3.connect(
        OUTPUT_DB
    )

    # --------------------------------------------------------
    # SQLite性能优化
    # --------------------------------------------------------

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA temp_store=MEMORY"
    )

    # --------------------------------------------------------
    # 重建数据库
    # --------------------------------------------------------

    rebuild_database(
        conn,
        table_schema
    )

    # --------------------------------------------------------
    # SQL验证
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("开始生成并验证SQL")
    print("=" * 70)

    sql_file = open(
        OUTPUT_SQL,
        "w",
        encoding="utf-8"
    )

    success_file = open(
        OUTPUT_SUCCESS,
        "w",
        encoding="utf-8"
    )

    error_file = open(
        OUTPUT_ERROR,
        "w",
        encoding="utf-8"
    )

    success_count = 0

    error_count = 0

    sql_start = time.time()

    for index, item in enumerate(
            train_data,
            1
    ):

        question = item.get(
            "question",
            ""
        )

        table_id = item.get(
            "table_id"
        )

        try:

            sql = generate_sql(
                item,
                table_schema
            )

            execution = execute_sql(
                conn,
                sql
            )

            result = {
                "question": question,
                "table_id": table_id,
                "sql": sql,
                "success": execution[
                    "success"
                ],
                "result": execution[
                    "rows"
                ],
                "columns": execution[
                    "columns"
                ],
                "error": execution[
                    "error"
                ]
            }

            line = json.dumps(
                result,
                ensure_ascii=False,
                default=json_default
            )

            sql_file.write(
                line + "\n"
            )

            if execution["success"]:

                success_file.write(
                    line + "\n"
                )

                success_count += 1

            else:

                error_file.write(
                    line + "\n"
                )

                error_count += 1

            # ------------------------------------------------
            # 每1000条打印一次
            # ------------------------------------------------

            if (
                    index % 1000 == 0
                    or index == len(train_data)
            ):
                elapsed = (
                        time.time()
                        - sql_start
                )

                speed = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"[SQL] "
                    f"{index}/{len(train_data)} "
                    f"| 成功={success_count} "
                    f"| 失败={error_count} "
                    f"| {speed:.1f} 条/秒"
                )

        except Exception as e:

            result = {
                "question": question,
                "table_id": table_id,
                "sql": None,
                "success": False,
                "result": [],
                "columns": [],
                "error": str(e)
            }

            line = json.dumps(
                result,
                ensure_ascii=False
            )

            error_file.write(
                line + "\n"
            )

            error_count += 1

    sql_file.close()

    success_file.close()

    error_file.close()

    conn.close()

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    total_elapsed = (
            time.time()
            - total_start
    )

    print()
    print("=" * 70)
    print("全部完成")
    print("=" * 70)

    print(
        f"表数量: "
        f"{len(table_schema)}"
    )

    print(
        f"问题数量: "
        f"{len(train_data)}"
    )

    print(
        f"SQL成功: "
        f"{success_count}"
    )

    print(
        f"SQL失败: "
        f"{error_count}"
    )

    print(
        f"总耗时: "
        f"{total_elapsed:.2f} 秒"
    )

    print()
    print("输出:")

    print(
        f"  {OUTPUT_DB}"
    )

    print(
        f"  {OUTPUT_SQL}"
    )

    print(
        f"  {OUTPUT_SUCCESS}"
    )

    print(
        f"  {OUTPUT_ERROR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
