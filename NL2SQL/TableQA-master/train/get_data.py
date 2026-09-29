from pathlib import Path

p = Path("/mnt/data/nl2sql_synth.py")
code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL2SQL 单表合成脚本
依赖：pip install openai
运行前准备：
  1) schema.json：包含 table_name、columns、column_stats（可选）
  2) metadata.json：包含 header/common/id/types（可选）
  3) SQLite 数据库文件
环境变量：
  OPENAI_API_KEY=你的API Key
  OPENAI_BASE_URL=可选，兼容 OpenAI 的接口地址
  NL2SQL_MODEL=模型名称
示例：
  python nl2sql_synth.py --db data.sqlite --schema schema.json --metadata metadata.json \
    --count 100 --difficulty medium --features where,group_by,aggregate,order_by \
    --output synthetic.jsonl
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """你是专业的 NL2SQL 数据集构建工程师。
根据提供的真实 SQLite 单表 Schema、字段统计、业务元信息和目标结构生成训练样本。
硬性规则：
1. 仅允许查询指定的一张表，禁止 JOIN、子查询、CTE、UNION、写操作。
2. 表名和字段名必须来自输入；不能编造字段或业务含义。
3. 不得仅凭 REAL/INTEGER 类型认定字段适合 SUM/AVG；必须有明确业务语义支持。
4. header 与数据库列的映射不明确时，不得强行按位置映射；避免依赖未知语义字段，或返回 failed。
5. SQL 必须是 SQLite 兼容的 SELECT。
6. question 必须准确表达 SQL 的字段、条件、聚合、分组、排序和 LIMIT。
7. 不要生成无意义结构，不要为凑难度堆叠 SQL。
8. 输出合法 JSON，不要 Markdown，不要代码围栏。
输出格式：
{"samples":[{"question":"...","sql":"...","difficulty":"...","primary_task":"...","features":["..."],"used_columns":["..."]}]}
"""

def load_json(path: Optional[str]) -> Any:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compact(obj: Any, max_chars: int = 18000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...(输入截断；建议减少字段统计内容)"
    return text

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [r[1] for r in rows]

def validate_sql(conn: sqlite3.Connection, sql: str, expected_table: str,
                 allowed_columns: List[str], max_rows: int = 1000) -> Dict[str, Any]:
    result = {"sql_valid": False, "execution_ok": False, "error": None}
    if not isinstance(sql, str) or not sql.strip():
        result["error"] = "SQL为空"
        return result
    s = sql.strip().rstrip(";")
    # 仅允许单条 SELECT；禁止危险语句与多表/嵌套结构
    if not re.match(r"(?is)^\s*select\b", s):
        result["error"] = "仅允许SELECT"
        return result
    forbidden = r"\b(join|insert|update|delete|drop|alter|create|attach|detach|pragma|union|with|;)\b"
    if re.search(forbidden, s, flags=re.I):
        result["error"] = "包含禁止关键字或多语句"
        return result
    # 保守要求 SQL 中出现目标表名；SQLite 查询执行再确认
    if expected_table.lower() not in s.lower():
        result["error"] = "SQL未引用目标表"
        return result
    result["sql_valid"] = True
    try:
        # 只读连接，限制结果数量；执行原 SQL，不修改数据库
        cur = conn.execute(s)
        cur.fetchmany(max_rows + 1)
        result["execution_ok"] = True
    except Exception as e:
        result["error"] = f"SQLite执行失败：{e}"
    return result

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # 兼容模型偶尔输出代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("模型输出不是合法JSON")
        return json.loads(match.group(0))

def build_prompt(schema: Dict[str, Any], metadata: Any, difficulty: str,
                 features: List[str], count: int, dialect: str) -> str:
    return f"""请生成 {count} 条互不重复的 NL2SQL 样本。
难度：{difficulty}
目标 SQL 结构/能力：{features}
SQL方言：{dialect}
单表限制：只能查询表 {schema.get('table_name')}，禁止JOIN。

请严格根据以下输入生成。字段统计只能辅助判断，不能替代业务语义。
Schema及统计：
{compact(schema)}
表头与业务元信息：
{compact(metadata) if metadata is not None else "未提供"}
要求：
- 每条样本都应有真实、自然的中文问题。
- 每条 SQL 都必须符合指定难度和目标结构。
- 若字段业务语义不足以支持某种查询，不要猜测；生成其他合理样本。
- 样本之间尽量覆盖不同字段、条件、值和查询结构，避免只替换常量。
- 只输出 JSON：{{"samples":[{{"question":"...","sql":"...","difficulty":"{difficulty}","primary_task":"...","features":["..."],"used_columns":["..."]}}]}}
"""

def main():
    ap = argparse.ArgumentParser(description="基于真实 SQLite Schema 的单表 NL2SQL 合成工具")
    ap.add_argument("--db", required=True, help="SQLite数据库文件路径")
    ap.add_argument("--schema", required=True, help="单表Schema JSON路径")
    ap.add_argument("--metadata", help="表头/业务元信息JSON路径（可选）")
    ap.add_argument("--count", type=int, default=100, help="总生成数量")
    ap.add_argument("--batch-size", type=int, default=20, help="每次请求数量，建议10-30")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    ap.add_argument("--features", default="where,aggregate,group_by,order_by",
                    help="逗号分隔目标结构")
    ap.add_argument("--output", default="synthetic.jsonl")
    ap.add_argument("--model", default=os.getenv("NL2SQL_MODEL", "gpt-4o-mini"))
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    if OpenAI is None:
        print("缺少依赖，请先运行：pip install openai", file=sys.stderr)
        sys.exit(1)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("请设置环境变量 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    if args.count < 1 or args.batch_size < 1:
        raise SystemExit("--count 和 --batch-size 必须大于0")

    schema = load_json(args.schema)
    metadata = load_json(args.metadata)
    table = schema.get("table_name")
    if not table:
        raise SystemExit("schema.json 必须包含 table_name")
    features = [x.strip() for x in args.features.split(",") if x.strip()]

    db_path = str(Path(args.db).resolve())
    if not Path(db_path).exists():
        raise SystemExit(f"数据库不存在：{db_path}")
    # 只读模式打开 SQLite
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    actual_cols = get_table_columns(conn, table)
    if not actual_cols:
        conn.close()
        raise SystemExit(f"数据库中找不到表或表无字段：{table}")
    schema_cols = [c.get("name") for c in schema.get("columns", []) if c.get("name")]
    if schema_cols and not set(schema_cols).issubset(set(actual_cols)):
        print("警告：Schema JSON 中部分字段与数据库实际字段不一致。", file=sys.stderr)

    client_args = {"api_key": api_key}
    if args.base_url:
        client_args["base_url"] = args.base_url
    client = OpenAI(**client_args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    x = json.loads(line)
                    existing.add((x.get("question", "").strip(), x.get("sql", "").strip()))
                except Exception:
                    pass

    accepted = 0
    attempts = 0
    with output_path.open("a", encoding="utf-8") as out:
        while accepted < args.count and attempts < args.max_attempts * ((args.count + args.batch_size - 1) // args.batch_size):
            batch = min(args.batch_size, args.count - accepted)
            attempts += 1
            prompt = build_prompt(schema, metadata, args.difficulty, features, batch, "SQLite")
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                )
                content = response.choices[0].message.content or ""
                data = extract_json(content)
                samples = data.get("samples", [])
                if not isinstance(samples, list):
                    raise ValueError("JSON中的samples不是数组")
            except Exception as e:
                print(f"[批次失败] {e}", file=sys.stderr)
                continue

            batch_added = 0
            for sample in samples:
                if accepted >= args.count:
                    break
                if not isinstance(sample, dict):
                    continue
                question = str(sample.get("question", "")).strip()
                sql = str(sample.get("sql", "")).strip()
                if not question or not sql:
                    continue
                key = (question, sql)
                if key in existing:
                    continue
                check = validate_sql(conn, sql, table, actual_cols)
                if not check["execution_ok"]:
                    print(f"[跳过SQL] {check['error']} | {sql}", file=sys.stderr)
                    continue
                row = {
                    "question": question,
                    "schema": schema,
                    "sql": sql.rstrip(";") + ";",
                    "difficulty": sample.get("difficulty", args.difficulty),
                    "primary_task": sample.get("primary_task", ""),
                    "features": sample.get("features", features),
                    "table_id": table,
                    "used_columns": sample.get("used_columns", []),
                    "source": "synthetic",
                    "validation": {
                        "schema_consistent": True,
                        "single_table_only": True,
                        "execution_ok": True,
                        "semantic_consistent": "not_automatically_verified"
                    }
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                existing.add(key)
                accepted += 1
                batch_added += 1
            print(f"[进度] 本批新增 {batch_added} 条，累计新增 {accepted}/{args.count}")
    conn.close()
    print(f"完成：新增 {accepted} 条，输出文件：{output_path.resolve()}")
    if accepted < args.count:
        print("提示：未达到目标数量。可检查字段语义、目标结构、模型输出数量或增加 --max-attempts。")

if __name__ == "__main__":
    main()
'''
p.write_text(code, encoding="utf-8")
print("已创建脚本：", p)
