# Enterprise Agent Workbench

The Enterprise Agent Workbench is an isolated application layer around the
existing Search-R1 capabilities. It demonstrates a merchant and advertising
operations workflow with explicit planning, structured tools, role-based
authorization, human approval, persistent threads, citations, streaming, and
an auditable trace.

This is an enterprise workflow prototype. The campaign APIs are deterministic
local mocks backed by SQLite; no real advertising account is connected or
modified. The workbench does not retrain Search-R1, rebuild its Retriever
index, or load vLLM into the FastAPI process. Model-dependent quality metrics
require a live A800 run and are not claimed by this repository implementation.

## Architecture

```text
Streamlit UI                  evaluation client
      |                              |
      +--------- HTTP / SSE --------+
                     |
                 FastAPI
                     |
          explicit LangGraph StateGraph
                     |
    +----------------+-------------------+
    |                |                   |
local tools     Search-R1 research   local vLLM HTTP
KB / analytics  E5+FAISS Retriever   Qwen Phase-3 actor
campaign mock   Phase-5 compressor   OpenAI-compatible API
    |                |                   |
SQLite demo DB  existing CPU server   separate GPU process
                     |
   SQLite checkpoints + explicit preference store + safe trace
```

LangGraph is only the orchestration layer. Search-R1 training, rollout,
reward, evaluation, and Phase-1 through Phase-6 benchmark behavior remain
unchanged.

The runtime separates an LLM data plane from a deterministic Agent control
plane:

- The **LLM data plane** interprets the task, proposes plans and replans, and
  synthesizes the final answer.
- The **deterministic Agent control plane** enforces canonical Tool IDs,
  Pydantic argument validation, role-aware Tool visibility, idempotency-key
  generation, PolicyEngine/RBAC decisions, side-effect detection, human
  approval, transactional write execution, audit logging, loop guards,
  checkpoint/resume, and citation validation.

The model proposes actions. The deterministic control plane authorizes and
executes them; security guarantees do not come from the LLM.

## Explicit graph

The application uses a raw `StateGraph`, with conditional edges rather than a
hidden high-level agent loop:

```text
START -> load_context -> planner -> policy_and_route
                                  |-- enterprise_kb_search --|
                                  |-- research_subgraph ------|
                                  |-- merchant_analytics -----+-> merge_evidence
                                  |-- campaign_read ----------|        |
                                  |                                    v
                                  |                                 replanner
                                  |                                    |
                                  +-- propose_business_write           |
                                              |                        |
                                     authorization_check               |
                                      | denied        | allowed         |
                                      v               v                 |
                               merge_evidence   human approval interrupt
                                                        |
                                                approve/edit/reject
                                                        |
                                             execute or return safely

policy_and_route -> finalizer -> citation_validation -> END
```

Maximum graph steps, tool calls, research searches, planner repairs, and
per-tool timeouts are configuration limits. A run cannot use an unbounded
Python loop. The planner stores only a concise `user_visible_reason`; hidden
chain-of-thought is neither requested nor retained.

## State contract

`AgentState` contains primitive, checkpoint-safe values:

- Identity: `thread_id`, `run_id`, `user_id`, `organization_id`, `role`.
- Request/context: `task`, `messages`, `conversation_summary`,
  `user_preferences`.
- Planning: `plan`, `next_action`, `action_arguments`, `pending_action`.
- Evidence: `tool_results`, `sources`.
- Control: `approval_request`, `approval_decision`, `step_count`,
  `tool_call_count`, `research_search_count`, `planner_repair_count`,
  `planner_failure_signatures`, `route`, `authorization_route`.
- Outcome: `errors`, `final_answer`, `final_citations`, `completed`,
  `termination_reason`, `citation_coverage`.
- Audit: `execution_trace`.

Reducers are used only for append-only messages, tool results, sources, trace
events, and errors. Replacement fields do not accidentally duplicate when a
checkpointed node is replayed.

Default safety limits are 40 graph steps, eight tool calls, two research
searches, one planner repair, a 20-second outer tool timeout, and conversation
summarization after 12 messages. They can be tightened with the corresponding
`WORKBENCH_MAX_GRAPH_STEPS`, `WORKBENCH_MAX_TOOL_CALLS`,
`WORKBENCH_MAX_RESEARCH_SEARCHES`, `WORKBENCH_MAX_PLANNER_REPAIRS`,
`WORKBENCH_TOOL_TIMEOUT_SECONDS`, and
`WORKBENCH_SUMMARY_MESSAGE_THRESHOLD` environment variables. Registry entries
also enforce their own lower per-tool timeouts.

## Model boundary

`WorkbenchModelClient` defines asynchronous `plan`, `replan`, `synthesize`,
and `summarize_memory` operations. Production uses
`VLLMHTTPModelClient`; CPU tests use the deterministic `FakeModelClient`.

Planner output is Pydantic-validated JSON containing an objective, one exact
registered action ID, arguments, a completed flag, and a user-visible reason.
Each call receives only the enabled, role-visible compact tool contracts plus
`finalizer`. Syntax/schema and semantic action/argument failures share one
deterministic repair request. A failed repair terminates as
`planner_parse_error`, `planner_semantic_error`, or `planner_stuck`; an action
description is never silently rewritten into an authorized tool choice.

Configure the separate model endpoint with:

```bash
export WORKBENCH_LLM_BASE_URL=http://127.0.0.1:8001/v1
export WORKBENCH_MODEL_NAME=phase3-search-r1
```

### Planner context optimization

The full Tool Registry schema exceeded the 8K deployment context before the
role-aware compact Planner catalog was introduced. These are measured prompt
construction results for the operator task `Increase C102 daily budget to
1200.`:

| Metric | Full Registry | Compact Planner | Change |
|---|---:|---:|---:|
| Catalog characters | 32,824 | 5,369 | -83.64% |
| Planner prompt characters | 33,897 | 7,647 | -77.44% |
| Planner prompt tokens | 9,698 | 1,790 | -81.54% |

Measured with the exact `Qwen2TokenizerFast` from the deployed checkpoint,
whose tokenizer-artifact fingerprint was
`7bdc8e14fc92822acfae6c899872a6d1ca27492d4017ac66b98bd26811c66f12`.
For the initial Planner request, the 8,192-token context minus the measured
1,790-token prompt, 700-token output reservation, and 256-token safety margin
leaves 5,446 tokens of headroom. Thus the measured initial Planner request is
safe for this 8,192-token deployment contract:

```text
8,192 - 1,790 - 700 - 256 = 5,446
safe_for_8192_initial_planner = true
```

The 1,790-token figure is specifically the initial Planner measurement.
Replanner requests can additionally include bounded prior tool results and
must not be described as always having that exact token count.
These measurements characterize prompt construction, not answer quality, EM,
latency, or production readiness. Business-state demonstrations continue to
use deterministic local SQLite fixtures.

### Engineering lessons

1. The full Registry schema produced a 9,698-token Planner prompt and overflowed
   the 8K context. A role-aware compact Planner catalog removed that overflow.
2. Free-form `next_action` values let descriptive strings such as `Run the
   structured merchant analytics operation update_campaign_budget...` reach
   policy and repeat as `unknown_tool`. Canonical IDs, semantic repair, and loop
   guards now stop that path safely.
3. This Workbench's async HITL path under Python 3.10 raised `Called get_config
   outside of a runnable context`. The validated deployment uses Python 3.11+
   for reliable interrupt/resume context propagation.
4. Long-context diagnosis exposed mismatched vLLM CUDA-graph capture limits.
   This deployment keeps `max_model_len` and `max_seq_len_to_capture` aligned.

## Tool Registry

Every tool has an explicit Pydantic input and output model, version, category,
risk, roles, timeout, retry/rate-limit configuration, read/write designation,
approval flag, idempotency declaration, source-production flag, and enabled
state. `GET /api/tools?tenant_id=...&role=...` returns only tenant-policy- and
role-visible safe metadata. Graph nodes call tools only through `ToolGateway`;
the Registry itself remains the protocol-neutral catalog.

| Category | Tools | Side effect |
|---|---|---|
| `knowledge` | `enterprise_kb_search` | None |
| `research` | `research_search` | One traced Retriever request |
| `analytics` | `campaign_performance_summary`, `compare_periods`, `channel_breakdown`, `conversion_funnel`, `roi_anomaly_detection`, `campaign_current_state` | Read-only parameterized SQL |
| `business_read` | `get_campaign`, `list_campaigns`, `get_budget_policy_status` | Read-only |
| `business_write` | `update_campaign_budget`, `pause_campaign`, `resume_campaign`, `create_followup_task` | Local SQLite write after approval |

The model never executes arbitrary SQL and cannot discover an unregistered
function.

## RBAC policy

| Role | Knowledge/research | Business reads | Analytics | Propose writes | Execute after approval |
|---|---:|---:|---:|---:|---:|
| viewer | yes | yes | no | no | no |
| analyst | yes | yes | yes | no | no |
| operator | yes | yes | yes | yes | yes |
| admin | yes | yes | yes | yes | yes |

All registered writes remain approval-required for operator and admin. The
planner supplies only business arguments; the graph derives a deterministic,
run-bound idempotency key before policy. Policy also checks that the tool
exists, is enabled, the run remains within budget, and a side-effecting action
carries that key. A denied tool never executes. If the same action receives the
same policy denial twice, the graph stops with `planner_stuck` instead of using
the framework recursion limit as control flow.

## Human approval lifecycle

1. The planner proposes an exact side-effecting tool ID and business arguments.
2. The graph validates the real tool schema and injects its deterministic
   idempotency key; policy then validates role, risk, budgets, schema, and key.
3. The graph includes current campaign state in the approval card when an
   earlier read made it available; the transactional API always records the
   authoritative before state at execution.
4. `interrupt()` returns a JSON-safe payload with action, arguments, reason,
   risk, requester, role, prior state, and allowed decisions.
5. No business write or premature audit write occurs before the interrupt.
6. Resume uses the same thread ID:
   - **approve** executes the original validated arguments;
   - **edit** validates the replacement JSON and executes only the edited
     arguments;
   - **reject** performs no write and records reviewer feedback.
7. The transactional business API records before/after state and an audit row.
   A unique idempotency key prevents double execution if resume replays.

## Search-R1 research integration

The research tool sends exactly one HTTP request represented by its trace
event to the configured existing Retriever:

```json
{"queries":["generated query"],"topk":3,"return_scores":true}
```

It validates the one-query result group and passes the scored nested passages
to `experiments/phase5_observation_context/evidence_compressor.py` through
`Phase5EvidenceAdapter`. The compressor is imported, not copied. It uses the
generated query only—never an answer or ground truth—and bounds the complete
Search-R1 information wrapper to the configured token budget (default 256).
The subgraph returns compressed evidence, source metadata, Retriever timing,
compression timing, and an explicit failure status.

Live mode requires `WORKBENCH_TOKENIZER_PATH` to name a local checkpoint
directory containing the exact Qwen tokenizer artifacts. It loads only the
tokenizer on CPU with local-files-only behavior and injects it through
`build_default_service` into the existing Phase-5 compressor; model weights are
not loaded in the Workbench process. Startup also verifies that the loaded
implementation is a Qwen2 tokenizer rather than labeling an arbitrary local
tokenizer as exact. Missing artifacts, an invalid path, a different tokenizer
family, or a load failure stops startup. The reversible lexical tokenizer is
available only when tests or local development explicitly set
`WORKBENCH_ALLOW_APPROX_TOKENIZER=true`; there is no silent live fallback.

`GET /healthz` reports the safe tokenizer mode, sanitized configured path,
tokenizer class, artifact fingerprint, compressor policy/version and
fingerprint, and maximum evidence-token budget. Evaluation fingerprints bind
these values together with model and Retriever configuration, so a partial run
cannot resume under a different token-accounting contract. The Retriever
remains a separate CPU service and vLLM remains a separate GPU service.

## Enterprise knowledge and merchant data

The enterprise fixture documents are searched by a deterministic BM25-like
lexical index; no new embedding model is required. Returned snippets are
bounded and carry stable IDs such as `KB-POLICY-001`.

`scripts/initialize_demo_data.py` creates deterministic local tables for
campaign state, daily metrics, orders, business audit events, and idempotency
records. Analytics operations use parameterized SQL and return derived metrics,
query identifiers, source identifiers, and measured latency. Business writes
are transactional and affect only this local database.

## Memory

Short-term state uses the LangGraph SQLite checkpointer keyed by `thread_id`.
It retains checkpoint-safe messages, plans, tool results, approval state,
trace, and final output so the exact thread can resume.

Long-term memory is a separate SQLite preference store namespaced by
`organization_id` and `user_id`. Only explicit allowlisted preferences are
accepted, such as reporting format, date range, KPI, and default merchant ID.
Credentials, access tokens, raw database rows, unrestricted tool output, and
hidden reasoning are rejected. Starting a new thread does not delete explicit
preferences; the API supports inspect, update, and delete operations.

When message count exceeds the configured threshold, the graph asks the model
client for a concise conversation summary. Original business evidence remains
in structured tool/source records rather than being copied into long-term
preference memory.

Set `LANGGRAPH_STRICT_MSGPACK=true` whenever SQLite checkpointing is used.

## Citations

Knowledge, research, analytics, and business-read tools return stable source
records. Final answers cite their IDs in markers such as `[KB-POLICY-001]`.
Citation validation rejects unknown IDs and the API returns sources separately
from answer text. Public projections omit internal paths and bound snippets.

Citation coverage is a deterministic lexical approximation:

```text
evidence-looking paragraphs or bullets containing citations
------------------------------------------------------------
all evidence-looking paragraphs or bullets
```

It measures marker coverage, not whether evidence semantically proves a claim.

## Trace and streaming

Trace events include `run_started`, node start/completion, planner decisions and
repairs, policy decisions, tool start/completion/failure, approval outcomes,
final answer creation, and run completion/failure. Events carry sequence,
timestamp, thread/run IDs, node/tool, non-negative duration, status, bounded
safe summaries, error type, and retry number.

Secret-looking fields, bearer credentials, and hidden-reasoning keys are
redacted or removed. Machine-readable JSON and a human-readable timeline are
available through the API. SSE streams graph progress without exposing model
reasoning tokens.

## HTTP API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/healthz` | Readiness and safe configuration summary |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/api/tools?tenant_id=...&role=...` | Tenant/role-visible Tool Registry metadata |
| GET | `/api/mcp/tools?tenant_id=...&role=...` | MCP-compatible read-only manifest |
| POST | `/api/mcp/call` | MCP-compatible read-only call adapter |
| POST | `/api/threads` | Create an identity-bound thread |
| POST | `/api/threads/{thread_id}/runs` | Start a business task |
| GET | `/api/threads/{thread_id}/stream` | Stream progress using SSE |
| GET | `/api/threads/{thread_id}/state` | Latest safe state projection |
| GET | `/api/threads/{thread_id}/history` | Checkpoint/node history |
| GET | `/api/threads/{thread_id}/trace` | Safe machine-readable trace |
| POST | `/api/threads/{thread_id}/resume` | Approve, edit, or reject |
| GET | `/api/users/{user_id}/memories` | Inspect explicit preferences |
| PUT | `/api/users/{user_id}/memories/{key}` | Add/update an allowlisted preference |
| DELETE | `/api/users/{user_id}/memories/{key}` | Delete one preference |

Pydantic request/response schemas form the service boundary. Raw LangGraph
state and implementation objects are never returned. Thread run/resume request
bodies and every thread inspection query require `tenant_id`; a tenant mismatch
returns the same 404 as a nonexistent thread.

## Streamlit workflow

The left panel selects user, organization, role, and thread. The main panel
accepts a task and shows final answer, citations, sources, and tool results. The
right panel shows the current plan, timeline, tool calls, latency, errors, and
an approval card with approve, reject, and validated JSON-edit controls.
Additional tabs show conversation memory, audit trace, and registry metadata.
The UI consumes only safe FastAPI responses.

## Isolated environment

Use Python 3.11 or newer. This requirement is scoped to reliable async
LangGraph interrupt/resume context propagation in this Workbench HITL
deployment; it is not a claim that every LangGraph use case requires Python
3.11. Do not install these dependencies into macOS base Python, the Search-R1
Conda environment, or `.venv-phase1`.

```bash
cd /Users/wangmingqi/Documents/search_rl
WORKBENCH_PYTHON="${WORKBENCH_PYTHON:-python3}"
command -v "${WORKBENCH_PYTHON}"
"${WORKBENCH_PYTHON}" --version
"${WORKBENCH_PYTHON}" -c \
  'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
"${WORKBENCH_PYTHON}" -m venv .venv-workbench
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
"${WORKBENCH_PYTHON}" -m pip install --upgrade pip
"${WORKBENCH_PYTHON}" -m pip install \
  -r apps/enterprise_agent_workbench/requirements-workbench.txt
export PYTHONPATH=/Users/wangmingqi/Documents/search_rl
export LANGGRAPH_STRICT_MSGPACK=true
```

Some pinned dependencies publish `Requires-Python` metadata that is compatible
with older interpreters, but the operational Workbench contract is stricter:
Python 3.10 fails before launch, while Python 3.11 and later pass the startup
check where the installed dependency set supports them. The launch scripts use
`WORKBENCH_PYTHON` rather than requiring a version-specific executable; they
print the resolved interpreter and version. No package installation is
performed by the repository scripts.

## Local CPU startup

Initialize the deterministic fixtures once:

```bash
cd /Users/wangmingqi/Documents/search_rl
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
export WORKBENCH_DATA_DIR=/Users/wangmingqi/.search_r1_workbench
"${WORKBENCH_PYTHON}" \
  apps/enterprise_agent_workbench/scripts/initialize_demo_data.py
```

With model and Retriever endpoints available, start the backend and UI in
separate terminals:

```bash
cd /Users/wangmingqi/Documents/search_rl
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
export WORKBENCH_LLM_BASE_URL=http://127.0.0.1:8001/v1
export WORKBENCH_MODEL_NAME=phase3-search-r1
export WORKBENCH_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
export WORKBENCH_ALLOW_APPROX_TOKENIZER=true
bash apps/enterprise_agent_workbench/scripts/run_api.sh
```

```bash
cd /Users/wangmingqi/Documents/search_rl
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
export WORKBENCH_API_BASE_URL=http://127.0.0.1:8010
bash apps/enterprise_agent_workbench/scripts/run_ui.sh
```

Open `http://127.0.0.1:8501`.

## A800 service startup

The commands below keep Retriever, model server, orchestration API, and UI in
four separate processes. Adjust only local asset paths to the audited A800
layout.

### 1. Existing CPU Retriever

```bash
cd /workspace/Search-R1
export PYTHONPATH=/workspace/Search-R1
export CUDA_VISIBLE_DEVICES=
python -m experiments.phase6_retriever_serving.optimized_retrieval_server \
  --index-path /workspace/searchr1-assets/wiki18/e5_Flat.index \
  --corpus-path /workspace/searchr1-assets/wiki18/wiki-18.jsonl \
  --model-path /workspace/searchr1-assets/models/e5-base-v2 \
  --index-backend flat \
  --faiss-thread-count 8 \
  --retrieval-encode-batch-size 32 \
  --top-k 3 \
  --cache-disabled \
  --host 127.0.0.1 \
  --port 8000
```

### 2. Trained Qwen vLLM server

Use the existing GPU environment containing vLLM 0.6.3:

```bash
cd /workspace/Search-R1
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20 \
  --served-model-name phase3-search-r1 \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-seq-len-to-capture 8192 \
  --gpu-memory-utilization 0.35 \
  --guided-decoding-backend lm-format-enforcer
```

The trained Phase-3 checkpoint is loaded from local disk with the XFormers
backend and offline Hugging Face/Transformers behavior. The compact Planner
catalog removed the original full-Registry 8K overflow, so a larger context is
not needed for that failure. For this vLLM deployment, the 8,192-token model
context and CUDA-graph capture boundary are explicitly aligned, and
`lm-format-enforcer` is used to avoid the incompatible Outlines path observed
in this environment. These settings are the final validated deployment
contract, not a general performance recommendation for other vLLM versions or
models. Do not substitute Outlines for this deployment.

### 3. LangGraph/FastAPI backend

```bash
cd /workspace/Search-R1
WORKBENCH_PYTHON="${WORKBENCH_PYTHON:-python3}"
command -v "${WORKBENCH_PYTHON}"
"${WORKBENCH_PYTHON}" --version
"${WORKBENCH_PYTHON}" -c \
  'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
"${WORKBENCH_PYTHON}" -m venv .venv-workbench
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
"${WORKBENCH_PYTHON}" -m pip install \
  -r apps/enterprise_agent_workbench/requirements-workbench.txt
export WORKBENCH_EVAL_RUN_ID="$(git rev-parse HEAD)-phase3-global_step_20-flat-wiki18-topk3-evidence256-merchant-seed-v1-run-001"
export WORKBENCH_DATA_DIR="/workspace/search_r1_workbench_evaluations/${WORKBENCH_EVAL_RUN_ID}"
if [ -e "$WORKBENCH_DATA_DIR" ]; then
  echo "Choose a new run identity and empty data directory"
  exit 1
fi
"${WORKBENCH_PYTHON}" \
  apps/enterprise_agent_workbench/scripts/initialize_demo_data.py \
  --overwrite
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WORKBENCH_TOKENIZER_PATH=/workspace/Search-R1/verl_checkpoints/phase3-qwen2.5-3b-small-real-grpo-training/actor/global_step_20
export WORKBENCH_LLM_BASE_URL=http://127.0.0.1:8001/v1
export WORKBENCH_MODEL_NAME=phase3-search-r1
export WORKBENCH_RETRIEVER_URL=http://127.0.0.1:8000/retrieve
"${WORKBENCH_PYTHON}" -m apps.enterprise_agent_workbench.tokenizer_preflight
bash apps/enterprise_agent_workbench/scripts/run_api.sh
```

The preflight prints only the tokenizer class, vocabulary size, pad/eos token
IDs, artifact fingerprint, and an exact-tokenizer-mode PASS marker. It does not
list tokenizer files or inspect model weights. `run_api.sh` repeats the same
fail-closed preflight before starting Uvicorn. On a network-isolated host,
prepare the venv from an audited local wheelhouse using pip's `--no-index` and
`--find-links` options; never install these packages into base Anaconda, the
Search-R1 Conda environment, or `.venv-phase1`.

Every new live evaluation must use a new run identity and a newly seeded,
dedicated `WORKBENCH_DATA_DIR`. The identity must encode the app revision,
model checkpoint, Retriever/index settings, relevant runtime overrides, seed,
and a unique run suffix. This prevents campaign writes from one evaluation
from contaminating another. For a continuation of an interrupted evaluation,
reuse both the exact identity and its unchanged data directory; do not reseed.

### 4. Streamlit

```bash
cd /workspace/Search-R1
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
export WORKBENCH_API_BASE_URL=http://127.0.0.1:8010
bash apps/enterprise_agent_workbench/scripts/run_ui.sh
```

Shell exports are terminal-local. The UI terminal must repeat its
`WORKBENCH_PYTHON` and API URL exports as shown; it does not inherit the API
terminal's environment. It does not need the tokenizer path because the UI is
an HTTP client of the already validated API.

### 5. End-to-end evaluation

This command records live model-dependent results; it has no fake-success mode:

```bash
cd /workspace/Search-R1
export WORKBENCH_PYTHON="$PWD/.venv-workbench/bin/python"
export PYTHONPATH=/workspace/Search-R1
export WORKBENCH_EVAL_RUN_ID="$(git rev-parse HEAD)-phase3-global_step_20-flat-wiki18-topk3-evidence256-merchant-seed-v1-run-001"
export WORKBENCH_DATA_DIR="/workspace/search_r1_workbench_evaluations/${WORKBENCH_EVAL_RUN_ID}"
"${WORKBENCH_PYTHON}" -m \
  apps.enterprise_agent_workbench.evaluation.run_evaluation \
  --api-base-url http://127.0.0.1:8010 \
  --run-config-identity "$WORKBENCH_EVAL_RUN_ID" \
  --fresh-isolated-database-confirmed \
  --output "${WORKBENCH_DATA_DIR}/evaluation/results.jsonl" \
  --overwrite

"${WORKBENCH_PYTHON}" -m \
  apps.enterprise_agent_workbench.evaluation.summarize_evaluation \
  --results "${WORKBENCH_DATA_DIR}/evaluation/results.jsonl" \
  --output-json "${WORKBENCH_DATA_DIR}/evaluation/summary.json" \
  --output-markdown "${WORKBENCH_DATA_DIR}/evaluation/summary.md"
```

The evaluation terminal must repeat `WORKBENCH_PYTHON`, `PYTHONPATH`, the exact
`WORKBENCH_EVAL_RUN_ID`, and its derived `WORKBENCH_DATA_DIR`; exports from the
API terminal are not shared. The evaluator obtains model, Retriever,
tokenizer, and compressor identity from the API's safe health metadata and
binds it into the resume fingerprint.

`--overwrite` replaces only the measured JSONL output; it does not reset the
SQLite campaign database. Never use it to rerun against a database already
mutated by another completed evaluation. Resume a partial result by omitting
`--overwrite` and retaining the same isolated database and run identity.

## Demo scenarios

The UI supports deterministic demonstrations of knowledge-only policy search,
C102 ROI comparison, mixed data-and-policy diagnosis, campaign reads, approved
budget update, rejected update, viewer denial, reviewer-edited update, explicit
preference memory, and traced tool failure recovery.

The example budget workflow is:

> Analyze why Campaign C102's ROI declined during the last seven days, check
> the relevant budget policy, recommend corrective actions, and increase the
> daily budget to 1200 if the change is compliant.

An operator run should pause before the local write. The database changes only
after an explicit approve or valid edit decision.

## Evaluation methodology

`evaluation/cases.jsonl` contains exactly 24 cases:

- 6 knowledge-only;
- 6 analytics-only;
- 6 mixed read-only;
- 6 write/approval/permission.

Cases declare expected tool categories, prohibited tools, approval and policy
behavior, source types, state deltas, and minimal acceptable facts. They do not
contain hidden answers, Retriever ground truth, or reward labels.

The live evaluator reports run completion, assertion-backed task completion,
routing, prohibited execution,
permission interception, approval/resume, idempotency, citation validity and
coverage, expected local database-state deltas, failure recovery, latency, tool
calls, planner repairs, and unfinished runs. Run completion means that the graph
finished with a nonempty answer and no recorded error; it is not itself a
semantic task-success claim. Task completion additionally requires every
applicable case assertion (facts, routing, source types, citations, policy,
approval, idempotency, and state delta) to pass. Fingerprints bind the case-file hash, API URL,
caller-supplied immutable run identity, and public server health/Tool Registry
metadata, so incompatible partial results are rejected. Generated
artifacts default to `~/.search_r1_workbench/evaluation` and are rejected if directed
inside Phase-4 or Phase-5 result directories.

Deterministic infrastructure and policy correctness belongs to the CPU test
suite. Model routing, synthesis, and answer quality belong to the live A800
evaluation. The summary never substitutes test-fixture outcomes for measured
model performance.

## Runtime platform components

### Observability

`RuntimeObservability` preserves the existing checkpoint-safe
`execution_trace` while adding real boundary spans and runtime metrics. The
span tree uses actual calls only: `Agent Run` contains `Planner`/`Replanner`,
`Tool/<name>` or `Retriever/research_search`, `LLM/<operation>`, and
`Finalizer`. If the OpenTelemetry SDK is installed, spans use an isolated SDK
provider. Setting `WORKBENCH_OTLP_ENDPOINT` enables the OTLP/HTTP exporter; an
absent SDK or collector never prevents local startup. Disable instrumentation
with `WORKBENCH_OBSERVABILITY_ENABLED=false`.

Prometheus scrapes `GET /metrics`. Dotted runtime names are exported with the
Prometheus-safe `workbench_` prefix and underscores:

| Runtime name | Prometheus family |
|---|---|
| `agent.run.latency` | `workbench_agent_run_latency` |
| `planner.latency`, `replanner.latency`, `llm.latency` | `workbench_planner_latency`, `workbench_replanner_latency`, `workbench_llm_latency` |
| `retrieval.latency`, `tool.latency` | `workbench_retrieval_latency`, `workbench_tool_latency` |
| `tool.calls`, `tool.errors`, `runtime.retries` | `workbench_tool_calls`, `workbench_tool_errors`, `workbench_runtime_retries` |
| `planner.repairs` | `workbench_planner_repairs` |
| `grounded_argument_bindings` | `workbench_grounded_argument_bindings` |
| `input_tokens`, `output_tokens`, `context_tokens`, `compressed_tokens`, `budget_headroom` | corresponding `workbench_*` families |
| `hitl.wait_time`, `checkpoint.count`, `resume.count` | corresponding `workbench_*` families |
| `run.success`, `run.failure` | `workbench_run_success`, `workbench_run_failure` |

`grounded_argument_bindings` uses only the low-cardinality `tenant_id` and
`argument_name` labels. Bound business values, thread IDs, and run IDs are never
included in this metric.

Minimal Prometheus scrape configuration:

```yaml
scrape_configs:
  - job_name: enterprise-agent-runtime
    metrics_path: /metrics
    static_configs:
      - targets: ["host.docker.internal:8010"]
```

Grafana can use that Prometheus data source directly. A starter dashboard can
graph the latency histograms, `rate(workbench_tool_errors[5m])`,
`rate(workbench_run_failure[5m])`, token counters, budget headroom, and HITL
wait time. Grafana is not required to run the workbench.

### ContextBudgetManager

Planner and Replanner input construction passes through
`ContextBudgetManager`. Its live tokenizer is the same injected exact Qwen
tokenizer used by the runtime; explicit test mode uses the deterministic local
tokenizer. Defaults are an 8,192-token context, 700-token generation reserve,
and 256-token safety margin. Component budgets are configurable:

```bash
export WORKBENCH_CONTEXT_TOKENS=8192
export WORKBENCH_GENERATION_RESERVE_TOKENS=700
export WORKBENCH_CONTEXT_SAFETY_MARGIN_TOKENS=256
export WORKBENCH_SYSTEM_CONTEXT_BUDGET=800
export WORKBENCH_TOOL_CATALOG_CONTEXT_BUDGET=1800
export WORKBENCH_MEMORY_CONTEXT_BUDGET=800
export WORKBENCH_TOOL_RESULTS_CONTEXT_BUDGET=2000
export WORKBENCH_CURRENT_TASK_CONTEXT_BUDGET=500
```

The deterministic policy keeps inputs already inside budget; truncates the
current task at a tokenizer boundary; summarizes memory by sorted allowlisted
preferences plus a bounded existing conversation summary; compresses tool
results to stable fields and bounded outputs; and drops oldest/overflow items
when no safe compact representation fits. It does not call another LLM.
Retrieved evidence remains compressed by the existing Phase-5 extractive
compressor before entering this manager. Each Planner state records decisions,
token counts, compressed tokens, and headroom. The historical initial-Planner
measurement remains 9,698 -> 1,790 exact tokens (-81.54%); it is documentation
evidence, not a hard-coded runtime result.

### Multi-tenant runtime namespace

The runtime namespace is `tenant_id / user_id / thread_id / run_id`.
`organization_id` is accepted only as a compatibility alias when creating a
thread and must equal `tenant_id` when both are supplied. Checkpoint keys prefix
the tenant, preference memory uses tenant/user keys, thread directory lookups
require tenant/thread, traces and business-write audit rows carry tenant ID,
and history/inspect/resume routes verify the tenant before checkpoint access.
Tenant tool policy deny rules override Registry and RBAC visibility. The
prototype still relies on caller-supplied identity; production authentication
must derive `tenant_id`, user, and roles from verified credentials.

### SDK and agentctl

The synchronous Python SDK defaults to `http://127.0.0.1:8010`; override it
with `AGENT_RUNTIME_BASE_URL`, `WORKBENCH_API_BASE_URL`, or the constructor:

```python
from apps.enterprise_agent_workbench.sdk import AgentRuntimeClient

with AgentRuntimeClient() as client:
    accepted = client.run(
        tenant_id="demo-org",
        thread_id="THREAD_ID",
        task="Inspect campaign C102",
    )
    state = client.inspect_thread(
        tenant_id="demo-org", thread_id="THREAD_ID"
    )
```

Expose the bundled script on `PATH` or invoke the module directly:

```bash
export PATH="$PWD/apps/enterprise_agent_workbench/scripts:$PATH"
agentctl --base-url http://127.0.0.1:8010 run \
  --tenant-id demo-org --thread-id THREAD_ID --task "Inspect campaign C102"
agentctl resume --tenant-id demo-org --thread-id THREAD_ID --decision approve
agentctl threads inspect --tenant-id demo-org --thread-id THREAD_ID
agentctl tools list --tenant-id demo-org --role viewer
agentctl eval run --run-config-identity RUN_ID \
  --fresh-isolated-database-confirmed
```

`AgentRuntimeClient` also exposes `resume`, `history`, `list_tools`, `health`,
and `metrics`.

### ToolGateway and MCP compatibility

`ToolGateway` is the only graph execution path. It composes Registry lookup,
Pydantic schema validation, version matching, RBAC, tenant allow/deny policy,
timeout, idempotent retry, an in-memory token bucket, tenant-bound idempotency,
safe audit hooks, and metrics. Side-effecting tools still require the existing
HITL approve/edit record, and timed-out synchronous writes retain the existing
transaction reconciliation behavior.

`MCPToolAdapter` translates internal `ToolSpec` metadata to an MCP-compatible
`tools` manifest and translates calls back through the same `ToolGateway`.
This release intentionally exposes read-only tools only. It is a compatibility
adapter, not a complete MCP server, transport implementation, or MCP
certification claim.

## Tests

Use the isolated workbench environment for application tests; no model download
or GPU is required:

```bash
cd /Users/wangmingqi/Documents/search_rl
PYTHONPATH=. ./.venv-workbench/bin/python -m pytest -q \
  tests/enterprise_workbench
```

Run existing Phase tests separately with `.venv-phase1`; the workbench never
installs LangGraph into that environment. Shell scripts are syntax-checkable
with:

```bash
bash -n apps/enterprise_agent_workbench/scripts/run_api.sh
bash -n apps/enterprise_agent_workbench/scripts/run_ui.sh
bash -n apps/enterprise_agent_workbench/scripts/agentctl
```

## Current limitations and production hardening

- Fixture data and business APIs are local deterministic mocks, not production
  integrations.
- The SQLite stores are suitable for a single-node demonstration; production
  needs managed transactional storage, migrations, backup, encryption, and
  concurrency/load testing.
- In-process run coordination and SSE need a durable queue/pub-sub layer for
  multiple API workers.
- Prometheus counters, local span records, rate-limit buckets, and Gateway
  idempotency cache are process-local; production multi-worker deployments need
  shared telemetry and quota/idempotency stores. OTLP export is optional.
- Authentication, SSO, tenant provisioning, key management, and network policy
  are outside this prototype. Caller-supplied IDs are not production identity.
- Policy rules are code-backed examples and need enterprise governance,
  versioning, and independent authorization review.
- The MCP layer is a read-only compatibility adapter, not a complete MCP
  transport/server or conformance certification.
- Retrieved open-domain evidence can be stale or misleading. Compression is
  lexical and citation presence does not prove support.
- Approximate lexical token accounting is intentionally restricted to explicit
  test/development mode. Live startup requires the local Qwen tokenizer and
  fails closed if its exact-tokenizer preflight cannot pass.
- The 24-case suite is a smoke evaluation, not a business-quality benchmark.
- Model-dependent quality, latency, and recovery results remain unmeasured until
  the documented A800 evaluation is run.
- Production writes require external idempotency, reconciliation, compensating
  actions, approval expiry, and tamper-evident audit retention.
