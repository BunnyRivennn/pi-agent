# pi-agent

Python tools for building AI agents and managing LLM deployments.

This repository is an open-source Python reimplementation of core ideas from Pi (JS/TS), starting with a provider-agnostic agent runtime that is composable and SDK-friendly.

## Status

Phase 1 and an initial Phase 2 slice are implemented:

- Core typed domain model for messages, content blocks, tools, usage, model config
- Async event stream primitive with terminal-result handling
- Agent loop runtime with turn/tool orchestration and steering/follow-up queues
- Stateful `Agent` wrapper with prompt/continue/abort/wait APIs
- Provider abstraction (`pi_ai`) with registry + runtime adapter
- Providers:
  - mock provider
  - OpenAI Responses provider (with streaming deltas)
  - OpenAI Completions-compatible provider (with streaming deltas)
- Strict checks and tests (`ruff`, `mypy`, `pytest`)

Roadmap details are tracked in [`PLAN.md`](./PLAN.md).

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

## Install

```bash
pip install pi-agent
```

```python
from pi_agent.agent_core import Agent, Model
```

## Development

```bash
uv sync
uv run ruff check .
uv run mypy src tests
uv run pytest -q
```

## Package layout

```text
src/pi_agent/agent_core/
  types.py         # domain model + event types + runtime protocols
  event_stream.py  # generic async stream + assistant stream specialization
  agent_loop.py    # turn execution + tool execution loop
  agent.py         # high-level agent state wrapper

src/pi_agent/pi_ai/
  types.py         # provider request + provider protocol
  registry.py      # provider registry + defaults
  runtime.py       # stream/complete APIs + Agent adapter
  providers/mock.py
  providers/openai.py
  providers/openai_completions.py

src/pi_agent/session/       # session persistence (checkpointer)
  types.py         # 4 entry types (message/compaction/branch_summary/custom)
  ids.py           # session UUIDv7 + entry ids
  values.py        # typed KV/list side-store addresses
  serialize.py     # entry <-> dict codecs
  errors.py        # SessionError / SessionCorruptError
  context.py       # compaction-aware context building
  session.py       # StorageBackedSession + MutationLine + branches
  repo.py          # SessionRepo (create/open/list/delete)
  factory.py       # AgentSession facade + create_agent_session

  storage/         # storage layer: contract + commit pipeline + backends
    storage_types.py # Storage protocol + Write union + scan queries
    commit.py        # seq allocation + write validation (backend-agnostic)
    memory.py        # InMemoryStorage
    sqlite.py        # SqliteStorage (WAL, transactional commit, seq counter)
```

Adding a backend means implementing `storage.Storage` and reusing
`commit.prepare_storage_commit` / `validate_committed_writes`; nothing outside
`session/storage/` needs to change.

## Session persistence example (no API key)

Run:

```bash
uv run python examples/session_demo.py
```

Demonstrates quit → restart → resume: a conversation (including a tool round) is
persisted to SQLite, the session is closed, then reopened by id into a brand-new
`Agent` with history and config restored. Replaying history does **not** re-run
tools — tool results are read back from storage.

Wiring it into your own code:

```python
from pi_agent.session import Context, SqliteSessionRepo, create_agent_session

ctx = Context()
repo = SqliteSessionRepo("~/.pi-agent/sessions")   # one .db file per session

# new session
sess = await create_agent_session(agent, repo, ctx)
session_id = sess.session.metadata.id

await agent.prompt("hello")
await agent.wait_for_idle()
await sess.flush()        # wait for in-flight persistence
await sess.dispose()

# later / another process: reopen by id, history is restored automatically
sess = await create_agent_session(agent2, repo, ctx, session_id=session_id)
```

Use `InMemorySessionRepo` for tests. Persistence is opt-in: an `Agent` without a
session behaves exactly as before.

## End-to-end example

Run:

```bash
uv sync --extra openai
export OPENAI_API_KEY=your_key_here
uv run python examples/agent_e2e.py
```

This example uses OpenAI (`gpt-5-mini`), routes calls through `pi_ai`, invokes a tool, and prints the final assistant response.

## Streaming example (OpenAI)

Run:

```bash
uv sync --extra openai
export OPENAI_API_KEY=your_key_here
uv run python examples/openai_streaming.py
```

This example prints streaming delta events (`text_delta`, and tool-call events when present) from the OpenAI provider.

## Build and publish

```bash
uv build
uvx twine check dist/*
uv publish
```

## Releases

Tag-based releases use `.github/workflows/release.yml` and expect:

- `pyproject.toml` version matches the tag (`vX.Y.Z`).
- A matching `CHANGELOG.md` section exists (`## [X.Y.Z] - YYYY-MM-DD`).
