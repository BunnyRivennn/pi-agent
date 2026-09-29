# GRPO 做 NL2SQL 完整技术笔记

> 基于当前主流工作：SQL-R1 / Reasoning-SQL / Arctic-Text2SQL-R1 / Think2SQL / AGRO-SQL 等

---

## 目录

1. [为什么 GRPO 适合 NL2SQL](#1-为什么-grpo-适合-nl2sql)
2. [GRPO 算法回顾](#2-grpo-算法回顾)
3. [整体训练范式：SFT 冷启动 + GRPO](#3-整体训练范式sft-冷启动--grpo)
4. [Reward 设计详解](#4-reward-设计详解)
   - [4.1 Format Reward](#41-format-reward)
   - [4.2 Syntax / Parse Reward](#42-syntax--parse-reward)
   - [4.3 Execution Reward（核心）](#43-execution-reward核心)
   - [4.4 Semantic / Structure Auxiliary Reward](#44-semantic--structure-auxiliary-reward)
   - [4.5 Penalty 机制](#45-penalty-机制)
   - [4.6 Reward 组合与权衡](#46-reward-组合与权衡)
5. [环境 / Rollout 搭建](#5-环境--rollout-搭建)
6. [数据构造](#6-数据构造)
   - [6.1 SFT 冷启动数据](#61-sft-冷启动数据)
   - [6.2 RL 采样池](#62-rl-采样池)
7. [推理阶段](#7-推理阶段)
8. [主要坑点与应对](#8-主要坑点与应对)
9. [面试一句话总结](#9-面试一句话总结)

---

## 1. 为什么 GRPO 适合 NL2SQL

NL2SQL 的任务特性天然适配 GRPO：

| 特性 | 说明 |
|------|------|
| **输出可验证** | SQL 可以在数据库中执行，得到明确的正确/错误信号 |
| **正确性不止于字符串匹配** | 需要执行结果等价或语义等价，而非简单的 Exact Match |
| **复杂推理链条** | Schema linking、多表 Join、嵌套、聚合、窗口函数、值条件推理 |
| **SFT 泛化不足** | SFT 容易记住格式和表层模式，复杂/未见过的查询泛化弱 |
| **有可执行环境** | 不需要训单独的 Reward Model，可用规则/验证信号 |

GRPO 相比 PPO 的优势：

- **无 Critic / Value 网络**：省显存，降低训练复杂度
- **组内相对 Advantage**：对同一 prompt 采样 G 条 SQL，用组内相对表现计算 advantage，更稳定
- **天然适合可验证奖励**：执行对/错、结果匹配、语法可解析等都可以规则化

---

## 2. GRPO 算法回顾

### 2.1 核心公式

对每个 prompt $q$：

- 采样 $G$ 个 output $o_1, o_2, ..., o_G$
- 每个 output 得到 reward $r_i$
- 组内归一化计算 advantage：

$$
A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G}) + \epsilon}
$$

- 优化目标（简化）：

$$
J_{\text{GRPO}} \approx \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \left(\min\left(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})\right)\right]
$$

其中：

- $\rho_i = \pi_\theta(o_i|q) / \pi_{\theta_{\text{old}}}(o_i|q)$：重要性采样比
- $\beta D_{\text{KL}}$：KL 散度惩罚，防止偏离 reference model 太远

### 2.2 关键参数经验值

| 参数 | 典型值 | 说明 |
|------|--------|------|
| G（组内采样数） | 8 ~ 16 | 太小 advantage 不稳定，太大成本高 |
| ε（clip 范围） | 0.2 | 标准 PPO clip |
| β（KL 系数） | 0.01 ~ 0.1 | 控制探索 vs 保持稳定性 |
| batch size | 64 ~ 256 | 视显存调整 |
| learning rate | 1e-6 ~ 5e-6 | 通常比 SFT 小 |

---

## 3. 整体训练范式：SFT 冷启动 + GRPO

### 阶段一：SFT Cold Start

**目的**：让模型学会基本格式、Schema 读取、简单 SQL 生成

**数据规模**：约 100K ~ 500K 条

**数据构成**：
- 简单单表查询
- 多表 Join 基础
- 常见聚合函数
- CoT 推理格式（Schema linking → Join path → Filter → Aggregation → SQL）

**示例格式**：

```
Question: 查询工资大于5000的员工姓名和部门名称

Schema:
- employees(id, name, salary, dept_id)
- departments(id, dept_name)

Thought:
1. 需要员工姓名(name)和部门名称(dept_name)
2. employees.dept_id = departments.id 建立连接
3. WHERE salary > 5000

SQL:
SELECT e.name, d.dept_name 
FROM employees e 
JOIN departments d ON e.dept_id = d.id 
WHERE e.salary > 5000
```

**效果**：SQL-R1 论文显示，用约 200K 合成数据冷启动 + 5K 复杂样本做 GRPO，效果能超过只 SFT 全量数据的方案。

### 阶段二：GRPO RL

**流程**：
1. 对每个 prompt 采样 G 个候选 SQL
2. 在沙箱数据库执行
3. 计算 reward（执行结果 + 辅助奖励）
4. 组内归一化 advantage
5. 更新策略

**训练技巧**：
- 先用简单样本 warmup，逐步增加难度
- 混合不同难度的样本，保持梯度多样性
- 定期评估 checkpoint，防止 reward hacking

---

## 4. Reward 设计详解

Reward 设计是整个系统的灵魂。单一的执行对/错会导致奖励稀疏，需要分层渐进设计。

### 4.1 Format Reward

确保输出格式符合预期。

| 条件 | 分值 | 说明 |
|------|------|------|
| 输出包含有效的 SQL 块 | +0.05 ~ 0.1 | 如 ```sql ... ``` |
| 不包含多余解释（若要求纯 SQL） | +0.05 | 避免推理时混入杂音 |
| 输出长度在合理范围内 | +0.05 | 防止退化输出 |
| 包含 Thought 格式（若启用 CoT） | +0.05 | 鼓励推理过程 |

### 4.2 Syntax / Parse Reward

确保 SQL 语法正确，减少无效探索。

| 条件 | 分值 | 说明 |
|------|------|------|
| SQL 能被 SQLGlot/sqlparse 解析 | +0.1 ~ 0.2 | 基本语法正确 |
| 所有表名存在于 schema 中 | +0.1 | 减少 schema 幻觉 |
| 所有列名存在于对应表中 | +0.1 | 同上 |
| JOIN 条件使用了存在的列 | +0.05 | 提高语义合理性 |
| 没有明显非法模式 | -0.1 ~ 0 | 空 SELECT、未绑定别名等 |

**注意**：语法对不代表执行对，但能显著减少无效探索。

### 4.3 Execution Reward（核心）

这是最主要的 reward 信号。

| 条件 | 分值 | 说明 |
|------|------|------|
| 执行成功（无报错） | +0.2 ~ 0.3 | 基础分 |
| 执行结果与 Gold SQL 完全一致 | +1.0 | 最高分 |
| 执行结果集合等价（order-insensitive） | +0.8 ~ 1.0 | 顺序不重要 |
| 执行结果部分匹配（F1 score） | 0 ~ 0.8 | 按比例给分 |
| 执行报错 | 0 | 不给负分，避免过于保守 |
| 执行超时/资源超限 | 0 或轻微负分 | 防止死循环 |

**实现细节**：
- 使用只读数据库快照（SQLite / Postgres read-only replica）
- 冻结时间函数（`NOW()`, `CURRENT_DATE` 等）
- 设置执行超时（如 10s）
- 结果缓存：相同 SQL + DB snapshot 可缓存
- 对浮点数比较设置容差（如 1e-6）
- 禁止 `DROP/INSERT/UPDATE/DELETE/ALTER/CREATE/ATTACH/PRAGMA` 等

### 4.4 Semantic / Structure Auxiliary Reward

辅助信号，帮助模型更快学到正确的结构模式。

| 条件 | 分值 | 说明 |
|------|------|------|
| 使用的表集合与 Gold 匹配 | +0.1 | 表选择正确 |
| JOIN 条件命中外键/关系图路径 | +0.1 | Join 逻辑合理 |
| 聚合函数与 Gold 一致 | +0.05 | COUNT/SUM/AVG 等 |
| GROUP BY 列与 Gold 一致 | +0.05 | 分组正确 |
| WHERE 条件中的列存在 | +0.05 | 过滤字段有效 |
| ORDER/LIMIT 结构与 Gold 一致 | +0.05 | 排序分页 |

**原则**：辅助奖励不超过总分的 30%，主奖励仍是执行等价。

### 4.5 Penalty 机制

| 行为 | 惩罚 | 说明 |
|------|------|------|
| 重复采样相同 SQL | -0.1 | 鼓励多样性 |
| 超出最大 token 长度 | -0.2 | 控制推理成本 |
| 使用未声明的表/列 | -0.1 | 减少幻觉 |
| 非只读语句 | -0.5 ~ -1.0 | 安全性 |
| 输出格式不符合要求 | -0.1 | 格式约束 |
| 同一个问题反复触发超时 | -0.2 | 防止恶意消耗资源 |

### 4.6 Reward 组合与权衡

**推荐组合方式**：

```
total_reward = w1 * exec_reward + w2 * syntax_reward + w3 * format_reward + w4 * structure_reward + penalty
```

**经验权重**：

| 分量 | 权重 | 说明 |
|------|------|------|
| Execution Reward | 0.6 ~ 0.8 | 主体 |
| Syntax / Parse Reward | 0.1 ~ 0.2 | 辅助 |
| Format Reward | 0.05 ~ 0.1 | 格式保证 |
| Structure Reward | 0.05 ~ 0.1 | 结构引导 |
| Penalty | 绝对值 < 0.2 | 约束 |

**注意事项**：
- 避免过度设计导致 reward hacking
- 定期分析 reward 分布，确保各分量尺度均衡
- 监控 KL 散度，防止策略突变

### 4.7 简单奖励

- 设计 GRPO 奖励函数：执行结果一致得 1.0 分，可执行但结果不一致得 0.1 分，其余 0 分；避免复杂奖励导致的 reward hacking，稳定提升模型执行准确率

---

## 5. 环境 / Rollout 搭建

### 5.1 Prompt 结构

每个 prompt 包含：

```
Question: {natural language question}

Database Schema:
{table_name(column1: type, column2: type, ...), ...}
Foreign Keys: {fk_constraints}
Sample Values: {optional, for value matching}

[Optional] Evidence: {additional context from BIRD dataset}

[Optional] Few-shot Examples: {2-3 examples}

Generate the SQL query.
```

#### 5.1.1 RAG知识库设计

```
{
  "type": "schema",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_1",
  "column_name": "序号",
  "synonyms": ["编号", "序号"]
}
###
{
  "type": "schema",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_5",
  "column_name": "产品名称",
  "synonyms": ["产品", "商品名称"]
}
###
{
  "type": "schema",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_6",
  "column_name": "规格型号",
  "synonyms": ["型号", "产品型号", "规格"]
}
###
{
  "type": "value",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_6",
  "column_name": "规格型号",
  "values": [
    "KH001",
    "KH002",
    "KH003",
    "...",
    "KH300"
  ]
}
###
{
  "type": "value",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_5",
  "column_name": "产品名称",
  "values": [
    "运动头盔",
    "自行车头盔",
    "摩托车头盔"
  ]
}
###
{
  "type": "value",
  "table_id": "ef611c00453d11e99316f40f24344a08",
  "column": "col_8",
  "column_name": "抽查结果",
  "values": [
    "合格",
    "不合格"
  ]
}
```

### 5.2 沙箱环境要求

| 组件 | 要求 |
|------|------|
| 数据库 | SQLite / PostgreSQL read-only replica |
| 连接池 | 复用，避免每次创建新连接 |
| 超时 | 单次执行 < 10s，总 rollout < 60s |
| 资源限制 | CPU/memory 配额，防止恶意查询 |
| 快照 | 固定数据集快照，保证可复现 |
| 缓存 | (question_hash, db_version, sql) → result |

### 5.3 Rollout 架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Prompt Pool │────▶│ Rollout Worker│────▶│  DB Sandbox  │
│  (questions) │     │ (G samples)  │     │ (read-only)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Reward Engine │
                    │ (rules + exec)│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ GRPO Update   │
                    │ (policy opt.) │
                    └──────────────┘
```

### 5.4 并发与效率

- 多进程 rollout worker，每个 worker 处理一批 prompt
- 每个 prompt 内 G 个采样串行或并行（取决于资源）
- 结果缓存减少重复执行
- 动态 batch size 适应显存

---

## 6. 数据构造

### 6.1 SFT 冷启动数据

**来源**：

| 数据集 | 用途 | 规模 |
|--------|------|------|
| Spider | 标准 NL2SQL 基准 | ~10K |
| BIRD | 更大规模、更真实 | ~12K |
| SQL-Creation | 合成数据 | ~50K |
| Synthetic | 自生成（GPT + schema） | 不限 |

**数据清洗规则**：
- 去除执行失败的 pair
- 去除字段幻觉严重的 pair
- 去除答案不稳定的 pair（多次执行结果不一致）
- 去除多义性问题

**难度分层**：

| 层级 | 特征 | 占比 |
|------|------|------|
| Level 1 | 单表，简单条件 | 30% |
| Level 2 | 多表 Join，复合条件 | 30% |
| Level 3 | 嵌套查询，聚合，GROUP BY | 20% |
| Level 4 | 窗口函数，外连接，复杂 CASE WHEN | 10% |
| Level 5 | 长上下文 schema，隐式关系推理 | 10% |

**CoT 格式要求**：

```
Thought:
1. Schema Linking: 识别问题中涉及的表和列
2. Join Path: 确定表之间的连接关系
3. Filter Condition: 解析 WHERE/Having 条件
4. Aggregation: 确定需要的聚合函数
5. Output Columns: 确定 SELECT 列

SQL:
SELECT ...
```

### 6.2 RL 采样池

**优先选择的样本类型**：

- ✅ 多表 Join（3张表以上）
- ✅ 隐式外键/需要推理的关系
- ✅ 嵌套查询 / EXISTS / IN / NOT IN
- ✅ GROUP BY + HAVING
- ✅ 排序分页与去重
- ✅ 数值聚合、日期处理
- ✅ 值条件需要匹配字符串/同义词/单位转换
- ✅ 长 schema（10+ 表）

**降低权重的样本**：
- ❌ 简单单表查询（模型已经接近 100% 正确）
- ❌ 重复模式过多的样本
- ❌ 答案不唯一/多义的样本

**采样策略**：
- 初始阶段：均匀采样各难度
- 中期：偏向模型当前表现较差的类别
- 后期：聚焦高方差样本（不同采样结果差异大的）

---

## 7. 推理阶段

### 7.1 推理流程

```
输入: Question + Schema
  │
  ├─→ 方案 A: Greedy Decoding
  │     输出单个 SQL
  │
  └─→ 方案 B: 采样 N 个候选
         │
         ├─→ 执行过滤（去掉执行失败的）
         ├─→ 结果一致性检查（取多数结果）
         └─→ 投票 / Self-Consistency 选最优
              │
              └─→ 输出最终 SQL
```

### 7.2 后处理

```python
def postprocess(raw_output: str) -> str:
    # 1. 提取 SQL 块（如果有 markdown 包裹）
    sql = extract_sql_block(raw_output)
    
    # 2. SQLGlot 格式化 + 标准化
    sql = sqlglot.transpile(sql, write='sqlite')[0]
    
    # 3. 安全检查：只允许 SELECT / WITH
    assert is_read_only(sql), "Non-SELECT statement detected"
    
    # 4. 移除注释（可选）
    sql = remove_comments(sql)
    
    return sql
```

### 7.3 生产部署考虑

| 方面 | 建议 |
|------|------|
| 延迟 | 单次推理 < 2s（不含执行） |
| 缓存 | (question_hash, schema_hash) → sql |
| 回退 | 如果执行失败，尝试简化/降级 |
| 监控 | 执行成功率、结果一致性、平均 token 数 |
| 安全 | 严格的 SQL 白名单，只允许 SELECT |
| 成本 | 控制采样数量，平衡质量与开销 |

---

## 8. 主要坑点与应对

### 8.1 Reward Hacking

**现象**：模型学会利用 reward 规则的漏洞，而非真正提升 SQL 质量。

**例子**：
- 输出固定格式但 SQL 本身脆弱/错误
- 针对特定数据库的特例优化
- 生成执行结果恰好匹配但语义不等价的 SQL

**应对**：
- 多样化数据库环境
- 使用语义等价/执行等价而非 Exact Match
- 定期人工抽检 reward 分布
- 引入对抗样本

### 8.2 执行非确定性

**现象**：相同 SQL 在不同时间执行结果不同。

**原因**：
- 时间函数（`NOW()`, `CURRENT_DATE`）
- 随机函数（`RAND()`, `RANDOM()`）
- 浮点聚合精度
- 未指定 ORDER BY 时的行序

**应对**：
- 冻结时间函数
- 设定随机种子
- 浮点数比较设置容差
- 强制 ORDER BY 或在结果比较时忽略顺序

### 8.3 奖励稀疏

**现象**：初期大部分采样都是错的，得不到正向 reward。

**应对**：
- 分层 reward（语法/格式/结构辅助）
- 课程学习（从简单到复杂）
- SFT 冷启动预热
- 适当降低执行对/错的门槛（部分匹配给分）

### 8.4 Schema 幻觉

**现象**：模型使用不存在的表名或列名。

**应对**：
- Syntax reward 中加入 schema 检查
- 在 prompt 中显式列出可用表和列
- 对幻觉字段给予负向惩罚
- 使用 schema retriever/RAG 在大 schema 场景下

### 8.5 长上下文 / 大 Schema

**现象**：schema 过大导致上下文溢出，或模型注意力分散。

**应对**：
- Schema 检索：只传入相关的表和列
- Schema 压缩：只保留类型和主外键
- 分段处理：先定位表，再写 SQL
- 使用长上下文模型（128K+）

### 8.6 训练 / 推理分布漂移

**现象**：RL 后模型推理风格变化（更长、更谨慎），导致延迟和成本上升。

**应对**：
- 监控平均推理 token 数
- 在 reward 中加入效率惩罚（如 token 数）
- 控制 KL 散度
- 定期评估推理成本

### 8.7 KL 控制不当

**现象**：
- KL 太大：模型偏离 ref 太远，格式崩坏
- KL 太小：模型学不动，进步缓慢

**应对**：
- 动态调整 β
- 监控 π_θ 和 π_ref 的输出分布
- 设置 KL 阈值告警

### 8.8 评测指标陷阱

**常见指标**：
- **EX (Execution Accuracy)**：执行结果匹配，最可靠
- **SM (Exact Set Match)**：结果集合等价
- **EM (Exact Match)**：字符串精确匹配，最严格但也最脆弱

**建议**：
- 主指标用 EX
- 辅指标用 SM
- 避免过度优化 EM（容易过拟合到特定写法）

---

## 9. 面试一句话总结

> **"NL2SQL 用 GRPO 的本质是把 SQL 生成变成可验证推理任务：SFT 冷启动保证格式和基础 Schema 能力，RL Rollout 对每个问题采样多条 SQL，在沙箱数据库执行，以执行等价为主、语法/格式/Schema 结构为辅助设计 Reward，GRPO 用组内相对 Advantage 更新，并加 KL 和格式约束。推理时多采样 + 执行验证。关键是奖励可验证、环境安全确定、避免 Schema 幻觉和 Reward Hacking。"**

**补充到 Agent 应用语境**：
> "NL2SQL 也可看成 Tool-using Agent 的子能力：LLM 调用 SQL Executor Tool，GRPO/RL 优化的是工具调用轨迹和结果验证闭环，和 Agent 的 Memory/Tool Server/Audit/权限隔离是同一套工程边界。"

## 10. 课程学习

```
原始 4.1 万样本
        ↓
数据清洗 / 去重 / SQL执行验证
        ↓
按难度分层
        ↓
┌──────────────┬──────────────┬──────────────┐
│     Easy     │    Medium    │     Hard     │
│              │              │              │
│ 简单查询      │ 多条件        │ 多实体        │
│ 单字段        │ AND / OR     │ GROUP BY     │
│ 单值匹配      │ 多字段        │ 聚合          │
│ 简单过滤      │ 时间条件      │ 多条件组合     │
└──────┬───────┴──────┬───────┴──────┬───────┘
       ↓              ↓              ↓
       └──────────── SFT ────────────┘
                      ↓
                找出模型错误样本
                      ↓
              Hard / Error Dataset
                      ↓
                    GRPO
```
怎么筛高质量 SFT 样本
怎么定义 NL2SQL 难度
怎么自动发现模型经常错的样本
怎么构造 GRPO Hard Set
怎么保证 RAG Context 和训练/推理完全一致

---

## 参考文献

1. SQL-R1: https://arxiv.org/abs/2504.04699
2. Reasoning-SQL: https://arxiv.org/abs/2504.04700
3. Arctic-Text2SQL-R1: https://arxiv.org/abs/2505.20315
4. Think2SQL: https://arxiv.org/abs/2504.04702
5. AGRO-SQL: https://arxiv.org/abs/2504.04703
6. DeepSeek-R1: https://arxiv.org/abs/2501.12948 (GRPO 原始论文)
7. BIRD: https://arxiv.org/abs/2305.03111
8. Spider: https://arxiv.org/abs/1809.08887

---

> **最后提醒**：具体实现时请以最新论文和官方代码为准，上述内容基于截至 2025 年上半年的公开工作整理。


## 10. LoRA / QLoRA

微调的核心思想是在不改变预训练模型原有能力的前提下，让模型去适应新的任务。  
预训练模型的原始权重是 $W$。在加入 LoRA 模块后，新的权重变为：

$$
W' = W + \Delta W = W + A \times B
$$

### 10.1 保证训练起点与原模型完全一致（零残差原则）

为了让模型在训练的**第一步（Step 0）**的表现与**预训练模型完全一致（即不引入任何随机噪声破坏预训练好的特征提取器）**，我们需要确保初始时的 $\Delta W = 0$。

- 因为 $B$ 初始化为零矩阵（$B=0$），所以 $A \times B = A \times 0 = 0$。
- 此时 $W' = W$，**模型完全保留了预训练权重的特性，可以安全地从原始模型的性能点开始训练。**

### 10.2 为什么不是把 $A$ 初始化为零？

前向永远是 $\Delta W \cdot x = A \cdot (B \cdot x)$：

- 初始 $B=0$ → 初始 $Bx=0$，$\Delta W x=0$，不破坏原模型权重。

反向第一步：

- $B$ 的梯度：$\frac{\partial L}{\partial B} = A^T \cdot \left(\frac{\partial L}{\partial (Ax)}\right) = A^T \cdot \frac{\partial L}{\partial \Delta W} \cdot x^T$，因为 $A$ 随机非零，所以 $B$ 立刻获得非零梯度，第一轮就能更新 $B$。
- $A$ 的梯度：$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \Delta W} \cdot (Bx)^T$，初始 $B=0$ 所以 $Bx=0$，$A$ 初始梯度为 0，等第一步 $B$ 更新到非零后，$Bx \neq 0$，$A$ 就自动获得非零梯度开始更新，完美接力。

如果反过来 $A=0$：

- $\frac{\partial L}{\partial B} = A^T \cdot (\dots) = 0^T \cdot (\dots) = 0$，第一步 $B$ 完全得不到梯度，$A$ 先更新一轮后 $B$ 才能动 —— 本质只是交换了谁先更新的角色，但工程上我们把随机权重放在 $A$（输入侧的低秩矩阵）、$B$ 置零，能让初始增量的尺度更可控，符合预训练权重“小幅度微调”的原则，所以成为通用默认方案。

---

## PPO

- **Actor**：负责学习策略，给定当前状态，输出下一步行动的概率；
- **Critic**：负责学习价值函数，给定当前状态，输出采取下一步行动能带来的收益（价值）。

Critic 给 Actor 提供优势函数，Actor 在环境中探索、交互、收集经验，这些经验又被用来训练 Critic 模型，使价值估计更加准确。

### Actor 的目标

在不过度偏离旧策略的前提下，尽量增大高优势动作的概率。

### Critic 的目标

让价值估计 $V(s)$ 尽量接近真实回报 $R_t$：

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]
$$

### PPO 的总损失函数（目标函数）

把 Actor / Critic / Entropy / KL 拼起来：

$$
\boxed{
\mathcal{L}_{\text{PPO}} =
\underbrace{\mathcal{L}_{\text{actor}}}_{\text{策略}}
+ c_v \underbrace{\mathcal{L}_{\text{critic}}}_{\text{价值}}
- c_e \underbrace{\mathcal{L}_{\text{entropy}}}_{\text{探索}}
+ \beta \underbrace{\mathbb{E}[\text{KL}(\pi_\theta \| \pi_{\text{ref}})]}_{\text{约束}}
}
$$

#### （1）策略函数损失

PPO 的核心思想非常简单粗暴，但又极其有效：我允许你更新策略，但你不能“跑偏”太多！它不像 TRPO 那样通过 KL 散度来严格约束，而是通过一个更直接的方式——剪裁（Clipping），来限制新旧策略之间的比率。

损失函数：

$$
\mathcal{L}_{\text{actor}} =
-\mathbb{E}_t \left[
\min\Big(
r_t(\theta) \hat{A}_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\Big)
\right]
$$

策略比率：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

两个策略的概率比。

#### 优势函数

现在，有了价值函数和 Q 函数，优势函数 $A^{\pi}(s,a)$ 就可以定义为：

$$
A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)
$$

**直观理解：**

- $Q^{\pi}(s,a)$ 是在状态 s 下采取动作 a 的预期收益。（单个动作的平均）
- $V^{\pi}(s)$ 是在状态 s 下，按照当前策略 $\pi$ 随机选择动作的平均预期收益。（多个随机动作的平均）
- 所以，$A^{\pi}(s,a)$ 就表示：在状态 s 下，采取动作 a 比按照当前策略的平均水平，能多获得（或少获得）多少奖励。
- 如果 $A^{\pi}(s,a)>0$，说明动作 a 比平均水平好，值得鼓励。
- 如果 $A^{\pi}(s,a)<0$，说明动作 a 比平均水平差，应该避免。

Critic 给的是价值估计 $V(s)$，然后优势函数 $A(s,a)$ 是用 Critic 算出来的：

$$
A(s,a) = Q(s,a) - V(s)
$$

在 PPO 里几乎都用 GAE（Generalized Advantage Estimation）：

$$
\hat{A}_t = \delta_t + \gamma\lambda \delta_{t+1} + \dots
$$

其中：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

👉 所以：Critic 不直接“给优势”，而是给价值，优势是从价值里算出来的。

#### （2）Critic 函数的目标

Critic 的目标是让价值估计 $V(s)$ 尽量接近真实回报 $R_t$：

$$
\mathcal{L}_{\text{critic}} = \mathbb{E}_t \left[ (V_\theta(s_t) - R_t)^2 \right]
$$

其中：

$$
R_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

在 LLM 里，Critic 通常是一个 Value Head，输入是 hidden state，输出一个标量。

✅ Critic 和 Actor 共享 backbone，但 loss 是分开的。

#### （3）PPO 的目标函数通常还会加上一个熵（Entropy）奖励项

$$
L^{PPO}(\theta)=L^{CLIP}(\theta)-c_{1}L^{VF}(\theta)+c_{2}S(\pi_{\theta})(s)
$$

其中：

- $L^{VF}(\theta)$ 是价值函数误差项（通常是均方误差），用于训练评论家网络：

$$
L^{VF}(\theta)=(V_{\theta}(s_{t})-V_{t}^{target})^{2}
$$

这里的 $V_{t}^{target}$ 可以是蒙特卡洛回报（未来总奖励），或者是 GAE 计算出的优势加上当前价值估计。

- $S(\pi_{\theta})(s)$ 是策略 $\pi_{\theta}$ 在状态 s 下的熵。熵衡量了策略的随机性或不确定性。

$$
S(\pi_{\theta})(s)=-\sum_{a}\pi_{\theta}(a|s)\log\pi_{\theta}(a|s)
$$

- $c_{1}$ 和 $c_{2}$ 是超参数，用于平衡各个损失项的重要性。

**为什么需要熵奖励？**

在强化学习中，智能体需要不断地在探索（Exploration）和利用（Exploitation）之间做出权衡。

- **利用**：智能体根据当前学到的最优策略，选择它认为能获得最高奖励的动作。
- **探索**：智能体尝试一些它不确定结果的动作，以发现更好的策略或未知的奖励。

如果策略的熵太低，意味着策略变得过于“确定”，它总是选择相同的动作，即使这些动作可能不是全局最优的。这会导致智能体陷入局部最优，无法发现更好的策略。这就像一个餐馆老板，一旦他发现一道菜受欢迎，他就只买这道菜，不再尝试其他新菜品，最终可能错失做大做强的机会。

通过在目标函数中添加一个正的熵项（因为我们是最大化目标函数，所以是加号），PPO 鼓励策略保持一定的随机性，从而促进探索。这就像给餐馆老板一个“创新奖励”，鼓励他尝试新的食材和烹饪方法，推出新菜品，即使这些尝试不一定每次都成功。


---

## GRPO
## GRPO 损失/目标

对每个 prompt q，用旧策略采样 G 个回答 {o_1,...,o_G}，分别打标量奖励 {r_1,...,r_G}。

### 1）组内相对优势

序列级（整段共用一个优势）：

$$
\hat A_i = \frac{r_i - \operatorname{mean}(\{r_1,\dots,r_G\})}{\operatorname{std}(\{r_1,\dots,r_G\}) + \varepsilon}
$$

- 只做均值中心化（不除 std）也可，等价于 RLOO/REINFORCE-baseline 的变体；
- std 为 0 时加小 eps（如 1e-8）防止除零；
- 若奖励只在末 token，仍把 \hat A_i 赋给该序列所有 token。

token 级写全：

$$
\hat A_{i,t} = \hat A_i,\qquad
\rho_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})}
$$

### 2）Clip 代理目标（要最大化）

$$
\mathcal J_{GRPO}(\theta)=
\mathbb E_{\substack{q\sim P(Q),\\ \{o_i\}_{i=1}^G\sim\pi_{\theta_{old}}(\cdot\mid q)}}
\left[
\frac1G\sum_{i=1}^G \frac1{|o_i|}\sum_{t=1}^{|o_i|}
\min\Big(
\rho_{i,t}(\theta)\,\hat A_{i,t},\;
\mathrm{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\,\hat A_{i,t}
\Big)
-\beta\,\mathbb D_{KL}[\pi_\theta\|\pi_{ref}]
\right]
$$

### 3）训练用的“损失”（取负）

$$
\mathcal L_{GRPO}=-\mathcal J_{GRPO}
$$

即策略部分用 `-min(ratio*A, clip(ratio)*A)`，再单独加 KL 惩罚 `beta*KL`。

### 4）KL 项（GRPO 默认用低方差非负估计）

$$
\mathbb D_{KL}[\pi_\theta\|\pi_{ref}]
\approx
\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}
-\log\frac{\pi_{ref}(o_{i,t}\mid\cdot)}{\pi_\theta(o_{i,t}\mid\cdot)}
-1
$$

逐 token 算后平均；\cdot 表示 (q,o_{i,<t})。[1](@ref)

---

## 和 PPO 的差异（一句话）

- 无 critic、无 GAE：优势用同 prompt G 条回答的组内 mean/std 代替 V(s)；[14](@ref)
- clip 重要性比、min 取保守目标，与 PPO actor 一样；
- KL 不作为 reward 折扣项塞进优势，而是直接加到目标/loss 里；
- 可选加 entropy 项促进探索，但标准 GRPO 原始式不含 entropy。[19](@ref)

## 简化情况

若每个采样批次只做 1 次梯度更新，开始时 \pi_\theta=\pi_{old}，则 \rho_{i,t}=1，clip 不生效，目标退化为：

$$
\mathcal J \approx
\mathbb E\left[
\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_{t=1}^{|o_i|}
\hat A_i\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-\beta\,KL
\right]
$$

也就是“组内标准化奖励的加权策略梯度 + 参考 KL”。

## GRPO 优势分配机制说明

### 核心要点

- GRPO 使用**结果奖励**（对整个输出序列的单一标量奖励 $r_i$）来计算优势，而非逐 token 的密集奖励。
- 但策略更新的**动作粒度仍然是单个 token**（即每次决策生成一个 token），因此需要将序列级的优势值均匀分配给该序列的每一个 token 位置。

### 数学表达

对于 prompt $q$，采样 $G$ 个输出 $\{o_1,\dots,o_G\}$，得到奖励集合 $\{r_1,\dots,r_G\}$。  
第 $i$ 个输出 $o_i$ 的组内标准化优势为：

$$
\tilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \varepsilon}
$$

将该优势值赋给输出 $o_i$ 的每一个 token 位置 $t$：

$$
\hat{A}_{i,t} = \tilde{r}_i, \quad \forall t \in [1, |o_i|]
$$

### 为什么保留下标 $t$

虽然同一序列内所有 token 的优势值相同，但保留时间步下标 $t$ 是为了后续按 token 计算重要性采样比率：

$$
\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

以及进行 clip 操作时需要区分每个 token 的概率比，因此形式上仍需携带 $t$。

### 与 PPO 的区别

- PPO 使用 Critic 网络 + GAE 产生逐 token 不同的优势值；
- GRPO 完全舍弃 Critic，直接用组内统计量作为全局优势，再广播到各 token。

### 实践注意事项

- 若组内奖励标准差很小（例如所有回答得分接近），除以 std 会放大微小差异，可能导致训练不稳定。此时可考虑仅减均值（不做归一化），即 RLOO 风格。
- 通常会在分母加上一个小常数 $\varepsilon$（如 $10^{-8}$）防止除零。

---

## GRPO 问题与思考 — 重点提炼

### 1. 组大小 G 的选择
- G 越大，多样性越好，但需视任务而定（输出空间小的任务无需太大）。
- 组内全对或全错会使该 step 的训练失去意义。

### 2. 探索与熵坍塌
- 缺乏探索 → 输出多样性不足 → 模型分布变尖锐 → 熵降低（熵坍塌）。
- 可通过不同采样方法、温度超参数控制组内多样性。

### 3. 极度依赖奖励函数
- 去掉价值函数后，优势完全由奖励计算，奖励必须公平、全面。
- 简单任务易设计，复杂任务（如智能体轨迹）难以权衡。
- 模型可能钻奖励空子，生成高分但不合理的输出。

### 4. 序列奖励 vs Token 级动作的不匹配
- 奖励是整个序列的标量，但动作是逐 token 生成。
- 这种粒度不一致会影响训练效果，后续 GSPO 会专门解决。

### 5. 重要性采样存在的问题
- 公式：ρ = π_θ(o_{i,t} | ...) / π_{θ_old}(o_{i,t} | ...)
- 作用：用旧策略采样的轨迹更新当前策略时，校正分布偏移。
- 但 GRPO 中的重要性采样有缺陷，GSPO 会详细分析。

> 总结：GRPO 简化了 PPO（去掉了 Critic 和 GAE），但引入了奖励设计困难、动作-奖励粒度不匹配、重要性采样偏差等问题，后续工作（如 GSPO）旨在改进这些不足。

---

## 11. 数据处理

### 11.1 数据集整体划分

本项目数据集划分如下：

| 数据集   |     数量 | 用途                          |
| ----- | -----: | --------------------------- |
| Train | 40,000 | 用于 SFT 训练及 GRPO Hard Set 构建 |
| Val   |  4,000 | 用于模型评估、训练过程监控及参数选择          |
| Test  |  4,000 | 用于最终模型效果评估                  |

Train、Val 和 Test 应在训练前完成划分，并尽量避免重复问题或高度相似问题分布在不同数据集中。

其中，Train 用于模型训练；Val 用于比较 SFT 和 GRPO 阶段的模型表现；Test 仅用于最终评估，不参与训练、Hard Set 筛选或参数调整。

### 11.2 数据清洗与标准化

对原始数据进行清洗和标准化，保证问题、表结构和 SQL 标注能够正确对应。

1. 读取原始问题、SQL 标注及表结构数据，检查必要字段是否缺失、格式是否异常。
2. 根据 `table_id` 关联对应的表结构，无法匹配表结构的样本不进入训练集。
3. 检查 SQL 标注中的列索引、聚合操作、条件操作符及条件值是否有效。
4. 将 SQL 标注中的列索引转换为数据库实际字段名，如 `col_1`、`col_2`，不修改原始数据库表结构。
5. 对重复问题、异常标注、缺失字段等样本进行标记，避免错误数据进入训练流程。
6. 保留原始问题、表 ID、SQL 标注及转换后的 SQL，便于后续追溯和排查。

### 11.3 SQL 正确性验证

对清洗后的样本生成 SQL，并进行执行和结果校验。

1. 根据原始 SQL 标注生成对应 SQL。
2. 在对应数据库中执行 SQL，检查是否能够正常执行。
3. 对执行成功的 SQL，进一步检查查询结果是否符合问题意图。
4. 对执行失败、查询结果错误或无法确认正确性的样本进行标记，并记录原因。
5. **只有 SQL 执行成功且查询结果正确的样本，才进入高质量训练样本集。**

SQL 执行成功并不代表查询逻辑正确。对于无法自动判断结果是否符合问题意图的样本，应进行人工抽检或暂缓使用，不直接认定为高质量样本。

### 11.4 样本难度划分

对通过正确性验证的样本，按照 SQL 查询复杂程度划分为简单、中等和困难三类。

**简单样本**

* 单表查询。
* 无筛选条件或仅包含一个简单筛选条件。
* 查询字段较少。
* 不包含复杂聚合、分组或排序逻辑。

**中等样本**

* 包含多个筛选条件。
* 包含聚合计算，如 `SUM`、`COUNT`、`AVG`。
* 包含排序、多个查询字段或多个实体筛选。
* 需要组合多个 SQL 操作才能完成查询。

**困难样本**

* 多实体、多条件组合查询。
* 分组统计、复杂聚合或聚合与筛选组合。
* 多表关联、嵌套查询或复杂逻辑组合。
* 需要综合理解多个字段、指标和筛选条件才能生成正确 SQL。

初期采用规则进行难度划分。对于同时符合多个难度条件的样本，按照较高难度类别归类；后续结合模型实际表现调整分类规则。

### 11.5 SFT 数据集构建

SFT 阶段以高质量样本为基础，训练模型根据用户问题和相关 Context 生成正确 SQL。

#### 11.5.1 样本组成及比例

第一轮 SFT 训练建议采用以下难度比例：

| 样本难度 | 建议比例 |   目标数量 |
| ---- | ---: | -----: |
| 简单   |  40% | 16,000 |
| 中等   |  40% | 16,000 |
| 困难   |  20% |  8,000 |
| 合计   | 100% | 40,000 |

上述数量以 40,000 条高质量 Train 样本为前提。若实际某一难度类别的合格样本不足，则使用已有高质量样本，不通过复制样本强行补齐比例。

#### 11.5.2 训练样本格式

每条 SFT 样本包含：

* 用户问题。
* RAG 检索得到的相关表结构、字段信息及候选值。
* 对应的正确 SQL。

训练输入为“用户问题 + RAG Context”，训练输出为正确 SQL。

训练阶段使用的 Context 应与实际推理阶段的检索方式、内容组织和格式保持一致，避免模型训练时获得推理阶段无法获取的信息。

#### 11.5.3 SFT 训练方式

采用由易到难的训练方式：

1. 先使用简单样本学习基本字段识别、条件筛选和 SQL 生成。
2. 再加入中等样本，学习多条件、聚合及排序等常见查询。
3. 最后加入困难样本，提升复杂条件组合、分组统计和复杂查询能力。

SFT 训练完成后，在 Val 集上评估模型表现，并保存模型输出和评估结果，作为后续 GRPO 训练的对照基线。

### 11.6 GRPO Hard Set 构建

GRPO 阶段重点使用 SFT 后仍然容易出错的中高难度样本，通过执行结果奖励优化 SQL 生成能力。

#### 11.6.1 样本来源

GRPO 候选样本主要来自：

1. SFT 模型在 Train 集上的错误样本。
2. 模型在离线评测中经常出错的问题。
3. 模型在线上推理中经常出错的问题。
4. 多实体、多条件、聚合及分组统计等复杂查询。

线上错误样本在加入训练集前，需要确认问题、正确 SQL 和对应 Context 均可靠。

#### 11.6.2 样本组成及比例

第一轮 GRPO 训练建议采用以下难度比例：

| 样本难度 | 建议比例 |  目标数量 |
| ---- | ---: | ----: |
| 简单   |  10% |   500 |
| 中等   |  30% | 1,500 |
| 困难   |  60% | 3,000 |
| 合计   | 100% | 5,000 |

第一轮可先构建约 5,000 条 GRPO Hard Set。该数量为实验起点，最终以筛选后实际符合条件的高质量样本数量为准。

GRPO 以中等和困难样本为主，保留少量简单样本用于监控基础能力是否退化。若简单样本已经稳定掌握，可进一步降低其比例。

#### 11.6.3 Hard Set 筛选原则

进入 GRPO Hard Set 的样本应满足以下条件：

1. 正确 SQL 已经过验证。
2. SQL 能够执行，且执行结果符合问题意图。
3. 训练时所需的 RAG Context 完整。
4. 问题具有明确答案，不存在无法消除的语义歧义。
5. 模型在该问题上存在可改善的错误。

对于因表结构缺失、RAG 检索错误、数据标注错误或数据库异常导致的问题，应先修复数据或检索流程，不直接作为 GRPO 训练样本。

### 11.7 模型评估指标

SFT 和 GRPO 阶段使用统一的评估集和评估口径，重点关注 SQL 执行正确率及不同难度样本的表现。

#### 11.7.1 SQL 执行正确率

统计模型生成的 SQL 中，能够正确执行且查询结果符合标准答案的样本比例。

该指标作为 NL2SQL 模型的核心评估指标。仅 SQL 语法正确、但查询结果错误的样本，不计为正确。

#### 11.7.2 不同难度样本正确率

分别统计简单、中等和困难样本的 SQL 执行正确率，分析模型在不同查询复杂度下的能力表现。

通过难度分层结果判断模型的主要短板，避免整体正确率掩盖困难样本表现不足的问题。

#### 11.7.3 GRPO 前后效果对比

在相同 Val 集上分别评估 SFT 模型和 GRPO 模型，比较：

* 整体 SQL 执行正确率变化。
* 简单、中等和困难样本正确率变化。
* 各类错误数量及错误类型变化。

通过对比判断 GRPO 是否带来实际提升，并分析提升主要来自哪些难度类别。

#### 11.7.4 简单样本能力退化监控

重点监控 GRPO 训练前后简单样本的正确率变化。

如果困难样本正确率提升，但简单样本正确率明显下降，则需要调整 GRPO 样本比例、训练轮数或训练参数，避免模型在提升复杂查询能力的同时损失基础 SQL 生成能力。

### 11.8 数据集与训练结果管理

对每轮数据处理和训练结果进行记录，保证实验可追溯、可复现。

1. 记录每条样本的问题、表 ID、正确 SQL、执行状态、结果校验状态和难度类别。
2. 对未通过验证的样本记录失败原因，便于后续修复和分析。
3. 记录 SFT 与 GRPO 使用的数据版本、训练配置及模型版本。
4. 保存 SFT 和 GRPO 在 Val 集上的整体及分难度评估结果。
5. 使用独立 Test 集进行最终评估，报告最终 SQL 执行正确率及各难度类别表现。

通过上述流程，形成“高质量样本筛选—SFT 基础训练—错误样本分析—GRPO 强化训练—统一评估”的持续迭代闭环。


