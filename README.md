# Secure RAG

An assistant that answers staff questions from company documents, with citations, and never shows a document the person is not allowed to read. It also runs actions through an agent that needs a person's approval for anything sensitive.

Everything runs offline by default (no API key, no cost). Set `RAG_LLM=claude` to use Claude. For a real-world setup, `docker compose up` adds Postgres and a Qdrant server (see [Docker stack](#docker-stack-postgres--qdrant)).

## What it shows

| Feature | Where |
| --- | --- |
| Document ingestion, chunking, embeddings, vector DB | `ingest.py`, `embed.py`, `store.py` (Qdrant) |
| Hybrid search (vector + BM25, reciprocal rank fusion) | `store.py` |
| Role-based access between group companies | `identity.py` rule, enforced in the Qdrant filter, the BM25 index and again after fusion |
| Answers with verifiable source references | `answer.py`; each answer cites document and section |
| Prompt-injection protection | `guard.py`: quarantine at ingestion, sources wrapped as data, output checks |
| Leak detection | canary string in the restricted board memo; any answer containing it is blocked |
| Tool-using agent, workflow states, checkpoints, human approval | `agent.py` (LangGraph `interrupt`, SQLite checkpoints) |
| Evaluation: retrieval, citations, refusals, leaks, cost, latency | `evaluate.py`, `data/golden.json` |
| Users, roles, document access | `identity.py`; Postgres tables `users` and `documents` when `DATABASE_URL` is set |
| Audit log | Postgres `audit_log` (or `data/runtime/audit.jsonl`), one row per question and agent step |
| HTTP API | `api.py` (FastAPI): ask, start an agent run, approve or reject |
| Observability | `tracing.py` (Langfuse): one trace per question or agent step, with retrieval, model call (tokens, cost) and output check |

## Architecture

```
                 ┌──────────── ingestion ────────────┐
 docs (md/pdf) ──► parse ─► sections ─► chunks ─► injection screen ──► quarantine (review)
                                              │
                                              ▼
                              Qdrant (vector + company/roles payload)
                              BM25 index (same chunks)

 user + question
   │
   ▼
 hybrid search ── filter by company AND role inside each retriever
   │              (the model never sees a forbidden chunk)
   ▼
 RRF fusion ─► can_read() re-check ─► relevance floor ─► none? refuse, no model call
   │
   ▼
 LLM (Claude or offline extractive) gets <source> blocks marked as data
   │
   ▼
 output check: citations must be real sources, no restricted canary
   │
   ▼
 answer + citations  (or refusal)  ──► audit log

 agent: request ─► plan (model proposes) ─► authorize (code: allow-list, args)
                    ─► read-only? execute
                    ─► sensitive? interrupt ─► approver: not requester, has role,
                                               same company ─► execute ─► outbox
```

## Run

Needs [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest -q                 # 66 offline tests, under a second; 15 Postgres/Qdrant tests skip
uv run rag users                 # sample users, companies, roles
uv run rag ingest                # chunks indexed and quarantined
uv run rag ask --user sara "How many leave days can I carry over?"
uv run rag ask --user lina "What retention applies under the steel subcontract?"   # other company: refused
uv run rag eval                  # golden set report
```

Agent with approval:

```sh
uv run rag agent --user omar "Raise a purchase of AED 40,000 from Falcon Steel Fabrication LLC for steel beams"
uv run rag approve <run id> --approver omar     # own request: ignored, still waiting
uv run rag approve <run id> --approver lina     # other company: ignored, still waiting (the API answers 404)
uv run rag approve <run id> --approver faisal   # finance, same company: executed
```

Above AED 100,000 only an executive (`noura`) can approve, as the procurement policy says.

## Docker stack (Postgres + Qdrant)

```sh
docker compose up -d --build        # Postgres :5433, Qdrant :6333, API :8010
export DATABASE_URL=postgresql://rag:rag@127.0.0.1:5433/rag
export QDRANT_URL=http://127.0.0.1:6333
```

| Store | Holds |
| --- | --- |
| Postgres `users` | user, companies, roles; read on every request |
| Postgres `documents` | each document's company and roles; seeded once from front matter, then the source of truth |
| Postgres `audit_log`, `eval_runs` | audit rows; one row per `rag eval` |
| Postgres (LangGraph tables) | agent checkpoints; a paused approval survives restarts |
| Qdrant server | vectors; collection named by a hash of chunks and embedder, reused on restart |

`data/golden.json` is the test set, not searchable content. The searchable content is `data/docs/`.

`rag db setup` keeps live changes and is safe to repeat (the API runs it on start): it seeds users only into an empty table, so a deleted user stays deleted, and adds rows for new documents. `rag db reset` restores every seeded user's roles and every document's access, undoing all live changes.

To block a document, empty its roles (`UPDATE documents SET roles = '{}' WHERE doc_id = ...`). Deleting its row also blocks it, but the next start re-registers it from front matter.

`GET /health` reports `"postgres"` and `"qdrant"`. If either is `false`, that process uses built-in users or an in-memory index, and changes made in Postgres do not apply to it.

Walkthrough (the API reads the caller from `X-User`):

1. Sara and Lina ask the same kind of question. Sara gets a cited answer; Lina (other company) is refused.
2. `uv run rag role revoke sara employee` and `uv run rag role revoke sara hr`. Sara's next question is refused, with no restart or re-index. `docker compose restart api`: still refused.
3. Omar raises an AED 40,000 purchase. Omar and Lina cannot approve it. Restart the API; Faisal approves the same run.
4. `docker compose exec postgres psql -U rag -d rag -c "select event, user_id, ts from audit_log order by ts"`.
5. `uv run rag db reset`, then `uv run rag eval`: 17/17, 0 leaks, saved to `eval_runs`.

A document is readable only if both its front matter and its Postgres row allow it. Narrowing access in Postgres applies at once. Widening it in Postgres alone opens nothing; change the front matter too, then restart (the index rebuilds when chunks or tags change) and run `rag db reset` or update the row.

Integration tests (they truncate tables, so they use the separate `rag_test` database):

```sh
RAG_TEST_DATABASE_URL=postgresql://rag:rag@127.0.0.1:5433/rag_test \
RAG_TEST_QDRANT_URL=http://127.0.0.1:6333 uv run pytest -q     # 81 tests
```

Unit tests ignore `DATABASE_URL` and `QDRANT_URL`, so running them in a shell that exports them is safe.

## HTTP API

```sh
RAG_AUTH=header uv run uvicorn secure_rag.api:app --port 8010      # docs at http://127.0.0.1:8010/docs
```

| Method | Path | Body |
| --- | --- | --- |
| GET | `/health` | |
| GET | `/me` | |
| POST | `/ask` | `{"question": "..."}` |
| POST | `/agent/runs` | `{"request": "..."}` |
| POST | `/agent/runs/{run_id}/decision` | `{"decision": "approve"}` or `"reject"` |

The caller is named by the `X-User` header (`current_user()` in `api.py`). That lets anyone claim any user, so it is off unless `RAG_AUTH=header` is set; without it every request gets 401. Docker Compose sets it and binds the API to 127.0.0.1. Production replaces that one function with a verified Entra ID token. The body cannot change identity: extra fields are ignored, and the decision endpoint uses the caller as the approver.

```sh
curl -s localhost:8010/ask -H 'X-User: sara' -H 'Content-Type: application/json' \
  -d '{"question": "How many leave days can I carry over?"}'
```

## Tracing (Langfuse)

Off unless `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set (plus `LANGFUSE_BASE_URL` for a self-hosted Langfuse). Each answer returns its `trace_id`.

```
ask            user id, tags, refused, reason, cited doc ids, cost
├─ retrieve    chunk ids and fused scores (only chunks the user may read)
├─ generate    model, input/output tokens, cost, stop reason   (Claude mode)
└─ output_check  ok / reason
agent_start / agent_resume   run id, status, reason; nested plan and ask spans
```

Only this app's spans are exported. The Anthropic SDK also emits a span per call; exporting it would count each call's cost twice and add Anthropic account ids.

Traces hold ids, scores and costs, not question, document or answer text: people who read traces may not have the asking user's document permissions. `RAG_TRACE_CONTENT=full` adds text, for a private test instance only.

## Settings

| Env var | Default | Meaning |
| --- | --- | --- |
| `RAG_LLM` | `offline` | `claude` uses Claude for answers and planning (costs money) |
| `RAG_MODEL` | `claude-opus-5-5` | Claude model id |
| `RAG_EMBEDDER` | `hash` | `ollama` uses a local Ollama embedding model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama model |
| `RAG_DATA_DIR` | `data/runtime` | audit log, agent checkpoints, outbox (when Postgres is off) |
| `DATABASE_URL` | unset | Postgres for users, access, audit, checkpoints, eval runs |
| `QDRANT_URL` | unset (in-memory) | Qdrant server for persistent vectors |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | unset | turn tracing on |
| `RAG_AUTH` | `off` | `header` trusts the `X-User` header (local use only) |
| `RAG_TRACE_CONTENT` | `ids` | `full` puts question, prompt and answer text in traces |

Claude calls use structured JSON output, a server-side fallback when the model declines, and record tokens, cost and time.

## Sample users

| User | Company | Roles |
| --- | --- | --- |
| sara | Alpha Construction | employee, hr |
| omar | Alpha Construction | employee, procurement |
| faisal | Alpha Construction | employee, finance |
| lina | Beta Properties | employee, finance |
| noura | both | employee, executive |

All companies, people and documents are invented.

## Limits (what production would add)

- Identity: users are seeded sample users (in Postgres when configured); production reads Entra ID groups through Microsoft Graph and SharePoint item permissions at ingestion.
- The offline composer quotes sentences by keyword overlap. It can pick the wrong sentence from the right document (e.g. "What retention applies under the steel subcontract?" returns the delay clause). It exists so tests run free; use `RAG_LLM=claude` for real answers.
- Claude mode, measured 17 Nov 2026 (claude-opus-5-5): eval 17/17, 0 leaks, mean latency 1.4 s, 7,453 tokens, $0.045 for the whole set. It answers the retention question correctly.
- Hash embeddings are lexical; use Ollama, Azure OpenAI or another embedding model for real semantic search.
- The injection screen is pattern-based. It is one layer; the permission filter and the tool allow-list are what actually hold.
- Observability: unit tests use an in-memory exporter. Checked live on 17 Nov 2026 against a local Langfuse 4.72 server: traces, user ids, chunk ids, model, tokens and cost arrive; no question, prompt or document text. The audit log stays local JSONL.
- API identity is a header naming a sample user, off unless `RAG_AUTH=header`. No rate limiting.
- Decisions on an agent run: only the requester and eligible approvers see the run; anyone else gets 404. Decisions on one run are serialised (in-process lock plus a Postgres advisory lock), and the outbox records each run once.
- Amounts must be finite numbers; an unreadable amount needs an executive.
- Docker Compose runs locally with local credentials (`rag`/`rag`), bound to 127.0.0.1. No production deployment yet. The outbox stays a JSONL file.
- Two processes starting at once on an empty Qdrant can race while building the collection; start the API before running CLI commands on a fresh stack.
- The API container uses port 8010, the same as the local `secure-rag-api` launch config; stop one before starting the other.
