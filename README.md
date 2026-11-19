# Secure RAG

An assistant that answers staff questions from company documents, with citations, and never shows a document the person is not allowed to read. It also runs actions through an agent that needs a person's approval for anything sensitive, and a cross-department workflow (HR → procurement → finance) that turns a hiring request or a bill of quantities (BOQ) into a purchase order.

Everything runs offline by default (no API key, no cost). Set `RAG_LLM=claude` to use Claude. For a real-world setup, `docker compose up` adds Postgres and a Qdrant server (see [Docker stack](#docker-stack-postgres--qdrant)).

## What it shows

| Feature | Where |
| --- | --- |
| Document ingestion, chunking, embeddings, vector DB | `ingest.py`, `embed.py`, `store.py` (Qdrant) |
| Hybrid search (vector + BM25, reciprocal rank fusion) | `store.py` |
| Role-based access between group companies | `identity.py` rule, enforced in the Qdrant filter, the BM25 index and again after fusion |
| Answers with verifiable source references | `answer.py`; each answer cites document and section |
| Right sources, wrong answer: caught before the user sees it | `guard.py`: answers keep to the company the question names; every sentence is cited; every number must appear in its cited source or the question |
| Prompt-injection protection | `guard.py`: quarantine at ingestion, sources wrapped as data, output checks |
| Leak detection | canary string in the restricted board memo; any answer containing it is blocked |
| Tool-using agent, workflow states, checkpoints, human approval | `agent.py` (LangGraph `interrupt`, SQLite checkpoints) |
| Evaluation: retrieval, citations, sections, wrong facts, context recall, refusals, leaks, cost, latency | `evaluate.py`, `data/golden.json` |
| Users, roles, document access | `identity.py`; Postgres tables `users` and `documents` when `DATABASE_URL` is set |
| Audit log | Postgres `audit_log` (or `data/runtime/audit.jsonl`), one row per question and agent step |
| HTTP API | `api.py` (FastAPI): ask, start an agent run, approve or reject |
| Cross-department workflow with two approval gates | `workflow.py` (LangGraph), `systems.py`, `policy.py`; see [Workflow](#workflow-hr--procurement--finance) |
| BOQ parsing, unit normalising, catalogue matching, pricing | `boq_parser.py` (tabular, Docling, LlamaParse), `boq.py` |
| Browser page for the workflow | `GET /ui` (`static/workflow.html`, no build step, no CDN) |
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
   │              (the model never sees a forbidden chunk);
   │              a company named in the question narrows to that company
   ▼
 RRF fusion ─► can_read() re-check ─► relevance floor ─► none? refuse, no model call
   │
   ▼
 LLM (Claude or offline extractive) gets <source> blocks marked as data,
   │  each labelled with its company
   ▼
 output check: citations must be real sources, no restricted canary
   │
   ▼
 company check: no other company's rule for a named company; two companies
   │            and none named ─► the answer must name them, else "name the company"
   ▼
 grounding check: every sentence cited; every number in its cited source
                  ("three" = 3, "12:30" is one number); a number from the
                  question only beside one from the source
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
uv run pytest -q                 # 233 offline tests, about two seconds; 15 Postgres/Qdrant tests skip
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

## Workflow: HR → procurement → finance

Two ways in, one approval path. Every number comes from code and the files in `data/systems/`; the model never sets a quantity, price, vendor or approver.

```
 request (sara, hr)                        BOQ file (khalid, site engineer)
 "3 site engineers join on 1 Dec.          xlsx / csv / pdf
  Order laptops and safety kits."               │
        │                                       ▼
        ▼                                  boq_parse   rows ─► lines (Pydantic); planted
   understand   intake: offline regex             instructions flagged, never followed
        │        or Claude (JSON schema)          │
        ▼                                       ▼
   hr_check     systems.find_hires():      boq_match   units normalised (sqm, m², SQ.M ─► m2);
        │        confirmed, same company,          code scores each line against the catalogue
        │        start date matches                 │
        ▼                                       ▼
   procurement  catalogue codes,           boq_price   quantity × catalogue rate (Decimal);
        │        quantities from HR,               unmatched / weak ─► RFQ list, unpriced;
        │        unknown vendors refused           high-value lines ─► review queue
        └───────────────────┬───────────────────┘
                            ▼
              budget hold for the total (systems.reserve_budget) ── short? stop, no approvals
                            ▼
              gate 1: procurement (omar)          interrupt(); reject ─► hold released, outbox empty
                            ▼
              finance: re-check the hold; approver from policy.py
                 ≤ AED 100,000 ─► finance or executive
                 > AED 100,000 ─► executive only
                            ▼
              gate 2: finance (faisal) / executive (noura)
                            ▼
              execute: requester's role re-checked ─► hold becomes spend ─► outbox: purchase order (+ IT ticket)
```

Each approver must not be the requester, must have the gate's role, must work for the same company, and may approve only one gate. An ineligible decision is ignored and the run keeps waiting. Each step writes an audit line and a Langfuse span with ids, counts, amounts and statuses.

The policy citation is fetched through the same permissioned `ask()` as the viewer: Omar sees "Procurement Policy › Purchase thresholds"; Sara sees that a citation exists, not its text.

```sh
uv run rag workflow start --user sara "3 site engineers join on 1 Dec. Order laptops and safety kits."
uv run rag workflow approve <run id> --approver omar
uv run rag workflow approve <run id> --approver faisal          # done: purchase order + IT ticket
uv run rag workflow show <run id>                                # timeline; --user to view as someone else
uv run rag workflow resume <run id>                              # finish a run stopped by a crash or failed write

uv run rag workflow boq --user khalid data/samples/boq_villa_blockwork.xlsx
uv run rag workflow approve <run id> --approver omar
uv run rag workflow approve <run id> --approver noura           # AED 215,925.00: executive only

uv run rag workflow-eval                                         # 11 workflow cases, offline
```

Browser: `RAG_AUTH=header uv run uvicorn secure_rag.api:app --port 8010`, then open http://127.0.0.1:8010/ui. Pick a user, start a request or load a sample BOQ, and approve or reject as the selected user. `WALKTHROUGH.md` has a five-minute walkthrough.

### BOQ parsers

| `RAG_BOQ_PARSER` | Reads | Install |
| --- | --- | --- |
| `tabular` (default) | xlsx, csv | included |
| `docling` | pdf (and scans), local | `uv sync --extra docling` (first run downloads models) |
| `llamaparse` | pdf, cloud | `uv sync --extra llamaparse`, `LLAMA_CLOUD_API_KEY` |

An xlsx file is always read in a child process, whatever the parser setting: at most 20 MB unzipped, 500 characters per cell and 5 million characters of output, killed past 512 MB of memory or 30 seconds. At most two run at once per server process; a third upload waits up to 30 seconds, then gets a busy error. Linux enforces the memory limit inside the child; macOS does not allow that, so there it is checked every 50 ms and can be exceeded by about 150 MB before the child is stopped.

After installing an extra, use `uv run --no-sync ...`: a plain `uv sync` or `uv run` removes extras it was not asked for.

Check a parser against a file, with accuracy when expected lines are given:

```sh
uv run rag boq-check data/samples/boq_villa_blockwork.xlsx --expected data/samples/boq_villa_blockwork.labels.json
RAG_BOQ_PARSER=docling uv run --no-sync rag boq-check data/samples/boq_villa_blockwork.pdf --expected data/samples/boq_villa_blockwork.labels.json
```

Real BOQs go in `data/private/` (git-ignored). Runs on those files keep counts only in the audit log, traces and CLI output, never line text, and LlamaParse refuses them unless `RAG_ALLOW_CLOUD_PARSE=yes` is set for that run. Claude never sees their lines.

`python scripts/make_samples.py` rebuilds the invented samples in `data/samples/`.

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
5. `uv run rag db reset`, then `uv run rag eval`: 20/20, 0 leaks, 0 wrong facts, saved to `eval_runs`.

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
| POST | `/workflows` | `{"request": "..."}` (needs the hr role) |
| POST | `/workflows/boq` | multipart `file`: xlsx, csv or pdf, at most 2 MB, content must match the extension (needs the site_engineer role) |
| POST | `/workflows/{run_id}/decision` | `{"decision": "approve"}` or `"reject"` |
| GET | `/workflows/{run_id}` | the run as the caller may see it; 404 for anyone but the requester and eligible approvers |
| GET | `/ui` | the workflow page |

Uploaded BOQs are saved under a random name, parsed, and deleted before the response.

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
| `RAG_TRACE_CONTENT` | `ids` | `full` puts question, prompt and answer text in traces, and `rag eval` prints each refused draft and the number or sentence that failed (never for `data/private/` BOQs) |
| `RAG_BOQ_PARSER` | `tabular` | `docling` or `llamaparse` |
| `RAG_ALLOW_CLOUD_PARSE` | `no` | `yes` lets LlamaParse read a file from `data/private/` on this run |
| `LLAMA_CLOUD_API_KEY` | unset | LlamaParse key |
| `RAG_SYSTEMS_DIR` | `data/systems` | HR, catalogue and budget files |
| `RAG_TODAY` | today | fixed date (YYYY-MM-DD) for resolving "1 Dec" in tests |

With `RAG_LLM=claude` the workflow also uses Claude to read the request into fields (JSON schema; code validates every field against the HR file and the catalogue), to write a two-line note for each approver, and to suggest a canonical description for weak BOQ lines (code still scores and decides).

Claude calls use structured JSON output, a server-side fallback when the model declines, and record tokens, cost and time.

## Sample users

| User | Company | Roles |
| --- | --- | --- |
| sara | Alpha Construction | employee, hr |
| omar | Alpha Construction | employee, procurement |
| faisal | Alpha Construction | employee, finance |
| lina | Beta Properties | employee, finance |
| noura | both | employee, executive |
| khalid | Alpha Construction | employee, site_engineer |

All companies, people, documents, vendors, prices and BOQs are invented.

## Limits (what production would add)

- Identity: users are seeded sample users (in Postgres when configured); production reads Entra ID groups through Microsoft Graph and SharePoint item permissions at ingestion.
- The offline composer quotes sentences by keyword overlap and reads only the top-ranked document. It can pick the wrong sentence from the right document; the eval's `sections` field and the "correct when context has the facts" metric show it. It exists so tests run free; use `RAG_LLM=claude` for real answers.
- Claude mode, measured 17 Nov 2026 (claude-opus-5-5) on the earlier 17-case set: 17/17, 0 leaks, mean latency 1.4 s, 7,453 tokens, $0.045. Measured 18 Nov 2026 on the 20-case set with the company and grounding checks: 19/20, 0 wrong facts, 0 leaks, mean latency 1.5 s, 11,995 tokens, $0.070. The one failure is a false refusal: a sentence that compares a number from the question with a cited limit ("AED 30,000 is above that limit [S1]") has no source number beside it, so the grounding check refuses it. The check stays strict, because the same rule stops "up to AED 30,000 [S1]" when the source says 25,000.
- The grounding check catches invented numbers and uncited sentences. It does not catch a wrong reading of a number the source does say (a rule for AED 50,000 applied to AED 60,000), a flipped meaning without a number, or a number taken from another sentence of the same source ("60 percent" when the source says "60 days"); that needs an LLM judge per sentence. Number words run from zero to ninety. A sentence that repeats an amount from the question needs a number from its source beside it, so "AED 60,000 needs finance approval [S1]" is refused. Markers must be written one per bracket (`[S1][S2]`, not `[S1, S2]`).
- Company names are a fixed list in `identity.py` (a stand-in for the HR system). Full names match in any case; short names ("Alpha", "Beta") only capitalised and not before a hyphen. A named company scopes the search but does not count as a search term. A question that names only a company the user cannot read gets group-wide documents only.
- A user who works for two companies and names neither gets "Name the company in your question" whenever the sources come from both and the answer does not say whose rule each is. The offline composer never names companies, so offline this always asks; Claude is told to name them.
- Hash embeddings are lexical; use Ollama, Azure OpenAI or another embedding model for real semantic search.
- The injection screen is pattern-based. It is one layer; the permission filter and the tool allow-list are what actually hold.
- Observability: unit tests use an in-memory exporter. Checked live on 17 Nov 2026 against a local Langfuse 4.72 server: traces, user ids, chunk ids, model, tokens and cost arrive; no question, prompt or document text. The audit log stays local JSONL.
- API identity is a header naming a sample user, off unless `RAG_AUTH=header`. No rate limiting.
- Decisions on an agent run: only the requester and eligible approvers see the run; anyone else gets 404. Decisions on one run are serialised (in-process lock plus a Postgres advisory lock), and the outbox records each run once.
- Amounts must be finite numbers; an unreadable amount needs an executive.
- Docker Compose runs locally with local credentials (`rag`/`rag`), bound to 127.0.0.1. No production deployment yet. The outbox stays a JSONL file.
- Two processes starting at once on an empty Qdrant can race while building the collection; start the API before running CLI commands on a fresh stack.
- The API container uses port 8010, the same as the local `secure-rag-api` launch config; stop one before starting the other.
- Workflow systems are JSON files read through `systems.py`. Production swaps `find_hires`, `price_items` and the budget functions for HR and ERP APIs, and the outbox for real purchase orders and tickets. Budget holds live in `data/runtime/budget_ledger.json` (a stand-in for ERP commitments): pricing holds the total, a rejection or stop releases it, execute turns it into spend. A run left waiting at a gate keeps its hold; there is no expiry. A failure after the hold releases it; a hard kill leaves the hold, and `rag workflow resume` finishes the run. A spent run always reaches its purchase order on resume. The ledger is one file per `RAG_DATA_DIR`: processes that share Postgres checkpoints must also share that folder (production keeps holds in the ERP). Still open: the same hires can be equipped by two runs, and an order split into two runs under AED 100,000 each never reaches the executive. Production records ordered (hire, item) pairs in the ERP.
- The approval matrix is `policy.py`, from the invented procurement policy. The policy names no approver below AED 10,000, so finance approves there too. A real matrix comes from finance.
- Request reading offline is regex-based: it knows the request shapes in the tests and refuses unknown items rather than guessing.
- BOQ matching is term overlap against a small catalogue (threshold 0.8, sizes and units must agree). A line is not priced when it names a word that sets a sibling item apart ("solid" vs "hollow"), when two items tie, when it describes removal or excluded work, or when its quantity is not a plain number. Unmatched or weak lines go to RFQ, never priced. In Claude mode a model suggestion only adds a hint to the RFQ entry. A real rate card and a review of the threshold on real BOQs come first.
- Rate and amount columns in a BOQ are ignored; prices come only from the catalogue.
- The `/ui` page uses the `X-User` header like the rest of the API: local use only. A file uploaded through it runs as a normal (not counts-only) run, even if it came from `data/private/`; use the CLI for private files.
- Run checkpoints keep the priced, RFQ and review lines with their descriptions (counts-only runs included), on the local disk or in Postgres. Raw parsed lines are dropped after pricing.
- A policy citation is cached per run and viewer for the life of the process; a role revoked meanwhile still sees the cached text until restart.
- If a step fails between gates (crash, failed write), the run waits until someone runs `rag workflow resume <run id>`; nothing retries on its own and the API has no resume route.
