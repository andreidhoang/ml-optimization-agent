# ML Intern — Phân Tích Hệ Thống Từ First Principles

> Tài liệu này giải thích toàn bộ hệ thống từ góc nhìn kỹ thuật sâu.
> Mục tiêu: không chỉ hiểu **cái gì** mà phải hiểu **tại sao** mỗi quyết định thiết kế lại được đưa ra.

---

## Mục Lục

1. [ML Intern Là Gì?](#1-ml-intern-là-gì)
2. [Bản Đồ Kiến Trúc Toàn Hệ Thống](#2-bản-đồ-kiến-trúc-toàn-hệ-thống)
3. [Luồng Dữ Liệu — Một Request Đi Qua Hệ Thống](#3-luồng-dữ-liệu--một-request-đi-qua-hệ-thống)
4. [Agent Core — Trái Tim Của Hệ Thống](#4-agent-core--trái-tim-của-hệ-thống)
5. [Session — Container Trạng Thái](#5-session--container-trạng-thái)
6. [ContextManager — Bộ Nhớ Của Agent](#6-contextmanager--bộ-nhớ-của-agent)
7. [ToolRouter — Tay Của Agent](#7-toolrouter--tay-của-agent)
8. [DoomLoop Detector — Hệ Thống Phòng Vệ](#8-doomloop-detector--hệ-thống-phòng-vệ)
9. [Backend — API Gateway & Session Pool](#9-backend--api-gateway--session-pool)
10. [Frontend — SSE Bridge & React Layer](#10-frontend--sse-bridge--react-layer)
11. [Data Flywheel — Vòng Lặp Thu Thập Dữ Liệu](#11-data-flywheel--vòng-lặp-thu-thập-dữ-liệu)
12. [Security Layer — Redact, Auth, Quotas](#12-security-layer--redact-auth-quotas)
13. [Các Quyết Định Thiết Kế Quan Trọng](#13-các-quyết-định-thiết-kế-quan-trọng)
14. [Thứ Tự Đọc Codebase](#14-thứ-tự-đọc-codebase)

---

## 1. ML Intern Là Gì?

ML Intern là một **autonomous AI agent** được xây dựng bởi HuggingFace. Nó có thể tự nghiên cứu, viết code, và deploy các ML project bằng cách sử dụng toàn bộ HF ecosystem: Hub, Datasets, Training Jobs, Spaces, Docs, Papers.

### Hai chế độ triển khai, một agent core

```text
┌─────────────────────┐      ┌──────────────────────────────────────┐
│   CLI (Local Tool)  │      │   Web App (HuggingFace Space)        │
│                     │      │                                      │
│  $ ml-intern        │      │  https://huggingface.co/spaces/...   │
│  $ ml-intern "..."  │      │  React + Vite frontend               │
│                     │      │  FastAPI backend                     │
│  agent/main.py      │      │  Multi-tenant, nhiều users đồng thời │
└──────────┬──────────┘      └──────────────────┬─────────────────┘
           │                                     │
           └─────────────────┬───────────────────┘
                             │
                    CÙNG MỘT AGENT CORE
                    agent/core/agent_loop.py
                    agent/core/session.py
                    agent/core/tools.py
```

**Tại sao thiết kế hai surface chia sẻ một core?**

First principle: **Không lặp lại business logic**. Agent logic (loop LLM, execute tool, manage context) là phần khó nhất và có nhiều edge case nhất. Nếu CLI và Web có hai implementation riêng biệt, bất kỳ bug fix hoặc improvement nào cũng phải thực hiện hai lần. CLI chính là "reference implementation" — nếu nó hoạt động trên CLI, nó sẽ hoạt động trên Web.

---

## 2. Bản Đồ Kiến Trúc Toàn Hệ Thống

```text
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DEPLOYMENT SURFACES                                  ║
║                                                                              ║
║  ┌──────────────────────┐         ┌──────────────────────────────────────┐  ║
║  │  CLI  (agent/main.py)│         │       Web App (HF Space)             │  ║
║  │                      │         │                                      │  ║
║  │  PromptSession       │         │  React + Vite + TypeScript           │  ║
║  │  (prompt_toolkit)    │         │  useChat (Vercel AI SDK)             │  ║
║  │        │             │         │  SSEChatTransport (custom bridge)    │  ║
║  │  submission_queue    │         │         │ POST /api/sessions/{id}    │  ║
║  │  event_queue         │         │         │ SSE stream response        │  ║
║  └──────────┬───────────┘         └─────────┼────────────────────────────┘  ║
║             │                               │                               ║
║             │                    ┌──────────▼────────────────────────────┐  ║
║             │                    │  FastAPI Backend  (backend/)           │  ║
║             │                    │                                        │  ║
║             │                    │  ┌─────────────────────────────────┐  │  ║
║             │                    │  │  SessionManager                 │  │  ║
║             │                    │  │  ├─ MAX_SESSIONS: 200           │  │  ║
║             │                    │  │  ├─ MAX_PER_USER: 10            │  │  ║
║             │                    │  │  ├─ sessions: dict[id, AgentSess│  │  ║
║             │                    │  │  └─ EventBroadcaster (fan-out)  │  │  ║
║             │                    │  └────────────────┬────────────────┘  │  ║
║             │                    │  Auth: HF OAuth   │  Quotas: Redis-free│  ║
║             │                    └───────────────────┼────────────────────┘  ║
║             │                                        │                       ║
╠═════════════╪════════════════════════════════════════╪═══════════════════════╣
║             │        AGENT CORE  (agent/)            │                       ║
║             ▼                                        ▼                       ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  submission_loop()  [agent_loop.py]                                  │   ║
║  │                                                                      │   ║
║  │  Đọc Operations từ submission_queue:                                 │   ║
║  │  USER_INPUT → Handlers.run_agent()                                   │   ║
║  │  EXEC_APPROVAL → resume sau approval                                 │   ║
║  │  INTERRUPT → session.cancel()                                        │   ║
║  │  COMPACT → _compact_and_notify()                                     │   ║
║  │  UNDO → context_manager.undo_last_turn()                             │   ║
║  │  SHUTDOWN → thoát vòng lặp                                           │   ║
║  └──────────────────────┬───────────────────────────────────────────────┘   ║
║                         │                                                   ║
║  ┌──────────────────────▼───────────────────────────────────────────────┐   ║
║  │  Handlers.run_agent()   — VÒNG LẶP AGENTIC CHÍNH                    │   ║
║  │                                                                      │   ║
║  │  Session                                                             │   ║
║  │  ├─ ContextManager ── message history + auto-compaction              │   ║
║  │  ├─ ToolRouter ──────── built-in tools + MCP servers                 │   ║
║  │  ├─ Config ──────────── model, yolo_mode, quotas                     │   ║
║  │  └─ logged_events ───── trajectory for SFT data collection           │   ║
║  │                                                                      │   ║
║  │  ┌──────────────────────────────────────────────────────────────┐   │   ║
║  │  │  AGENTIC LOOP (max 300 iterations per turn)                  │   │   ║
║  │  │                                                              │   │   ║
║  │  │  1. compact check (nếu > 85% context window)                 │   │   ║
║  │  │  2. doom_loop check (detect A,A,A / [A,B,A,B] patterns)      │   │   ║
║  │  │  3. with_prompt_caching(messages, tools)                     │   │   ║
║  │  │  4. litellm.acompletion() — streaming hoặc batch             │   │   ║
║  │  │  5. emit assistant_chunk events → event_queue                │   │   ║
║  │  │  6. if no tool_calls: emit turn_complete, DONE               │   │   ║
║  │  │  7. for each tool_call:                                       │   │   ║
║  │  │       a. _needs_approval()? → emit approval_required         │   │   ║
║  │  │          wait EXEC_APPROVAL operation                        │   │   ║
║  │  │       b. tool_router.execute_tool()                          │   │   ║
║  │  │       c. context_manager.add_message(result)                 │   │   ║
║  │  │  8. goto 1                                                   │   │   ║
║  │  └──────────────────────────────────────────────────────────────┘   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                         TOOL ECOSYSTEM                                       ║
║                                                                              ║
║  sandbox_tool ──── remote code exec (HF Space)                              ║
║  research_tool ─── multi-step sub-agent với dedicated LLM calls             ║
║  jobs_tool ──────── HF Training Jobs (GPU clusters)                         ║
║  docs_tools ──────── HF documentation search + fetch                        ║
║  papers_tool ──────── ArXiv papers                                          ║
║  dataset_tools ─────── HF Hub datasets inspection                          ║
║  web_search ──────────── Tavily web search                                  ║
║  plan_tool ────────────── Structured planning với step tracking             ║
║  notify_tool ─────────────── Slack/gateway out-of-band notifications       ║
║  hf_repo_files ──────────────── HF repo CRUD (read/write/delete)           ║
║  hf_repo_git ─────────────────────── Git operations trên HF repos          ║
║  MCP server (hf-mcp-server) ──── HF Hub native MCP tools                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                         DATA FLYWHEEL                                        ║
║                                                                              ║
║  session.logged_events ──► save_trajectory_local() ──► .tmp → atomic rename║
║       │                                                                      ║
║       └──► subprocess.Popen(session_uploader.py) ── detached, fire-forget  ║
║                │                                                             ║
║                ▼                                                             ║
║  smolagents/ml-intern-sessions (HF Dataset)                                 ║
║       │                                                                      ║
║       └──► scripts/build_sft.py ──► SFT training data                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Luồng Dữ Liệu — Một Request Đi Qua Hệ Thống

### 3a. Luồng Web (Browser → Frontend → Backend → Agent → SSE)

```text
Browser                Frontend               Backend              Agent Core
  │                       │                      │                      │
  │  User gõ message      │                      │                      │
  │──────────────────────►│                      │                      │
  │                       │ POST /api/sessions   │                      │
  │                       │ /{id}/submit         │                      │
  │                       │ {text: "..."}        │                      │
  │                       │─────────────────────►│                      │
  │                       │                      │ submit_user_input()  │
  │                       │                      │─────────────────────►│
  │                       │                      │                      │ submission_queue
  │                       │                      │                      │ .put(USER_INPUT)
  │                       │                      │                      │
  │                       │ SSE stream opens     │                      │ submission_loop
  │                       │◄─────────────────────│                      │ dequeues
  │                       │                      │                      │
  │                       │ data: {processing}   │◄─────────────────────│ event_queue.put()
  │                       │◄─────────────────────│ EventBroadcaster     │
  │                       │                      │ fan-out to sub       │
  │  UI: "thinking..."    │ data: {assistant_    │                      │
  │◄──────────────────────│  chunk: "Tôi sẽ..."}│◄─────────────────────│ streaming tokens
  │                       │                      │                      │
  │  UI: renders text     │ data: {tool_call:    │                      │ execute tool
  │◄──────────────────────│  sandbox, code:..}   │◄─────────────────────│
  │                       │                      │                      │
  │  UI: shows tool card  │ data: {tool_output:  │                      │ tool returns
  │◄──────────────────────│  "output..."}        │◄─────────────────────│
  │                       │                      │                      │
  │  UI: final response   │ data: {turn_complete}│                      │ done
  │◄──────────────────────│◄─────────────────────│◄─────────────────────│
  │                       │                      │                      │
  │                       │ SSE stream closes    │                      │
```

### 3b. Luồng Approval (khi agent cần permission)

```text
Agent                  event_queue           Backend SSE            Frontend
  │                        │                     │                      │
  │  Tool cần approval     │                     │                      │
  │  (hf_jobs, sandbox,    │                     │                      │
  │   destructive ops)     │                     │                      │
  │────────────────────────►                     │                      │
  │  {approval_required,   │ EventBroadcaster    │                      │
  │   tools: [...]}        │────────────────────►│                      │
  │                        │                     │──────────────────────►
  │  session.pending_      │                     │  onApprovalRequired  │
  │  approval = tools      │                     │  callback fires      │
  │                        │                     │                      │
  │  PAUSED — chờ          │                     │                      │ User clicks
  │  EXEC_APPROVAL op      │                     │                      │ Approve/Deny
  │                        │                     │◄─────────────────────│
  │                        │                     │ POST /approve        │
  │◄────────────────────────────────────────────  │                      │
  │  submission_queue      │                     │                      │
  │  .put(EXEC_APPROVAL)   │                     │                      │
  │                        │                     │                      │
  │  RESUME — execute tool │                     │                      │
```

---

## 4. Agent Core — Trái Tim Của Hệ Thống

### File: `agent/core/agent_loop.py`

Đây là file quan trọng nhất của toàn bộ hệ thống. Nó chứa hai thành phần chính:

#### 4.1 `submission_loop()` — Control Plane

```text
submission_loop(session, submission_queue)
        │
        ▼
  while session.is_running:
        │
        ├── dequeue Operation (timeout=1.0s để check is_running)
        │
        ├── Op.USER_INPUT ──────► Handlers.run_agent(session, text)
        │
        ├── Op.EXEC_APPROVAL ───► resume_after_approval(session, approvals)
        │
        ├── Op.COMPACT ─────────► _compact_and_notify(session)
        │
        ├── Op.UNDO ────────────► context_manager.undo_last_turn()
        │                         + emit undo_complete event
        │
        ├── Op.SHUTDOWN ────────► session.is_running = False, break
        │
        └── Op.INTERRUPT ───────► session.cancel()
                                   (signal dừng giữa chừng)
```

**Tại sao dùng queue thay vì gọi hàm trực tiếp?**

Vấn đề: Agent đang ở giữa một LLM call (đang stream tokens) — lúc này user bấm Ctrl+C (interrupt). Nếu không có queue, ta phải xử lý interrupt bằng exception handling, signal, hoặc thread — tất cả đều phức tạp và dễ leak resource.

Với queue: INTERRUPT là một Operation được đưa vào queue. `submission_loop` nhận Op này, gọi `session.cancel()` (set asyncio.Event), và vòng lặp agentic kiểm tra `session.is_cancelled` sau mỗi iteration. **Clean cancellation không cần exception magic**.

Ngoài ra, queue còn cho phép toàn bộ control flow (approve, undo, compact) là **first-class operations** — không phải side-channel hacks.

#### 4.2 `Handlers.run_agent()` — Data Plane

```text
run_agent(session, text=None)
    │
    ├── Nếu có pending_approval và user gửi message mới:
    │   └── _abandon_pending_approval() — inject CANCELLED tool results
    │       để LLM context luôn hợp lệ (mỗi tool_call phải có tool_result)
    │
    ├── Thêm user message vào ContextManager
    │
    └── VÒNG LẶP (max 300 iterations):
            │
            ├── 1. _compact_and_notify()
            │      Kiểm tra: running_context_usage > compaction_threshold?
            │
            ├── 2. check_for_doom_loop(messages)
            │      Nếu detect loop → inject corrective prompt vào messages
            │
            ├── 3. with_prompt_caching(messages, tools, model_name)
            │      Anthropic models: thêm cache_control breakpoints
            │      Khác: pass-through không thay đổi
            │
            ├── 4. _call_llm_streaming() hoặc _call_llm_non_streaming()
            │      litellm.acompletion() với unified interface
            │      emit assistant_chunk events
            │
            ├── 5. Nếu finish_reason == "stop" và không có tool_calls:
            │      emit turn_complete, RETURN
            │
            ├── 6. Xây dựng tool_calls list từ accumulated deltas
            │
            ├── 7. Kiểm tra malformed JSON (LLM đôi khi gen sai JSON args)
            │      _detect_repeated_malformed() → inject error message
            │
            ├── 8. _needs_approval(tool_call, config)?
            │      Nếu Yes: emit approval_required, set pending_approval
            │               PAUSE — wait for EXEC_APPROVAL operation
            │      Nếu No:  tiếp tục
            │
            ├── 9. tool_router.execute_tool(name, args)
            │      emit tool_call event (trước khi execute)
            │      emit tool_output event (sau khi execute)
            │
            ├── 10. context_manager.add_message(tool_result)
            │       cập nhật running_context_usage
            │
            └── goto 1
```

**Tại sao `_abandon_pending_approval` inject CANCELLED results?**

Anthropic API yêu cầu: mỗi `assistant` message có `tool_calls` phải được theo sau bởi `tool` messages với đúng `tool_call_id`. Nếu user gửi message mới trong khi đang chờ approval, ta phải inject synthetic tool results với content "CANCELLED BY USER" — nếu không, API sẽ trả về lỗi về malformed conversation history.

---

## 5. Session — Container Trạng Thái

### File: `agent/core/session.py`

Session là **đơn vị cô lập** của mỗi conversation. Nó chứa tất cả state cần thiết để một agent hoạt động.

```text
Session
├── session_id: str (UUID)        ← định danh duy nhất cho tracing
├── config: Config                ← model, yolo_mode, save_sessions, ...
├── context_manager: ContextManager  ← toàn bộ conversation history
├── tool_router: ToolRouter       ← tool registry + MCP clients
├── event_queue: asyncio.Queue    ← output channel
├── _cancelled: asyncio.Event     ← interrupt signal
├── pending_approval: dict | None ← tool calls đang chờ user approve
├── sandbox: Sandbox | None       ← remote code execution space
├── _running_job_ids: set[str]    ← HF training jobs đang chạy
├── logged_events: list[dict]     ← trajectory cho data collection
├── turn_count: int               ← đếm số turns để auto-save
├── model_effective_effort: dict  ← cache kết quả probe effort cascade
└── notification_gateway: ...     ← Slack/webhook notifications
```

#### 5.1 Event System

```python
async def send_event(self, event: Event) -> None:
    # 1. Đưa event vào queue (để CLI/Web render)
    await self.event_queue.put(event)
    
    # 2. Log vào trajectory (cho data collection)
    self.logged_events.append({...})
    
    # 3. Auto-notification (Slack, webhook)
    await self._enqueue_auto_notification_requests(event)
    
    # 4. Heartbeat save (mid-turn, không block)
    HeartbeatSaver.maybe_fire(self)
```

**Tại sao một hàm làm 4 việc?** Vì mọi event đều cần cả 4 side effects này. Nếu tách ra, mỗi callsite trong `agent_loop.py` phải nhớ gọi đủ 4 — dễ bỏ sót. `send_event` là single point of truth.

#### 5.2 Trajectory Saving — Atomic Write Pattern

```text
save_trajectory_local():
    1. scrub() — xóa secrets (hf_token, API keys) khỏi payload
    2. Tính filepath (stable per session — không tạo file mới mỗi lần)
    3. Write to .tmp file (filepath + ".tmp")
    4. os.rename(.tmp → filepath)  ← atomic trên POSIX
    
    Tại sao atomic? Nếu process crash giữa chừng khi đang write,
    ta có file .json cũ (đầy đủ) thay vì file .json mới (bị truncate)
    mà retry scanner không đọc được.
```

#### 5.3 Detached Upload Pattern

```text
save_and_upload_detached(repo_id):
    1. save_trajectory_local()  ← fast, synchronous
    2. subprocess.Popen(
           [sys.executable, "session_uploader.py", "upload", path, repo_id],
           start_new_session=True,   ← detach khỏi parent process
           stdin/stdout/stderr=DEVNULL
       )
    
    Tại sao subprocess thay vì asyncio task?
    - asyncio task: nếu main process bị kill, task bị cancel
    - subprocess với start_new_session=True: tiếp tục sống sau khi
      parent chết. Upload không bao giờ bị mất dù server restart.
```

---

## 6. ContextManager — Bộ Nhớ Của Agent

### File: `agent/context_manager/manager.py`

```text
ContextManager
├── items: list[Message]           ← toàn bộ conversation [system, user, assistant, tool, ...]
├── model_max_tokens: int          ← context window của model (từ litellm.get_model_info)
├── running_context_usage: int     ← token count hiện tại (cập nhật sau mỗi LLM call)
├── compact_size: int              ← 10% của model_max_tokens = reserved space sau compaction
├── untouched_messages: int = 5    ← số messages gần nhất không bao giờ bị compact
└── system_prompt: str             ← từ agent/prompts/system_prompt_v3.yaml (Jinja2 template)
```

#### 6.1 Compaction Threshold

```text
model_max_tokens = 200,000  (Claude Sonnet, ví dụ)
compact_size     = 20,000   (10%)

compaction_threshold = 200,000 - 20,000 = 180,000 tokens

Khi running_context_usage > 180,000:
    needs_compaction = True
```

**Tại sao không compact tại 100%?** Nếu compact tại 100%, ta không còn đủ tokens để gọi LLM và xử lý kết quả compaction. Reserve 10% là buffer an toàn.

#### 6.2 Compaction Algorithm

```text
Trước compaction:
[system] [user_1] [assistant_1] [tool] [tool] [user_2] [assistant_2] ... [user_N] [last_5_msgs]
    │        │                                                               │          │
    │    first_user_msg (task ban đầu — không bao giờ compact)           kept      kept
    │
    └── preserved

Sau compaction:
[system] [user_1] [SUMMARY: "Agent đã làm X, Y, Z vì..."] [user_N] [last_5_msgs]

SUMMARY được tạo bằng cách gọi LLM với prompt đặc biệt:
"Tóm tắt conversation trên, tập trung vào key decisions, WHY, 
 problems solved, context cần thiết cho người mới."
```

**Tại sao giữ `user_1` (first user message)?** Đây là task ban đầu. Agent cần luôn nhớ mình đang làm gì. Mất task ban đầu = agent không biết mình đang làm gì.

**Tại sao giữ 5 messages cuối?** Để agent không bị "mất mạch" giữa chừng của một operation. Nếu đang execute một sequence of tool calls, 5 messages cuối đảm bảo context gần nhất luôn đầy đủ.

#### 6.3 Dangling Tool Call Patch

```text
_patch_dangling_tool_calls():
    Vấn đề: Anthropic API yêu cầu mỗi tool_call trong assistant message
    phải có một matching tool_result message. Trong quá trình compaction
    hoặc undo, một số tool_result có thể bị xóa nhưng tool_call vẫn còn.
    
    Fix: Scan toàn bộ items, tìm tool_calls không có matching tool_result,
    inject synthetic tool_result với content "TOOL_RESULT_MISSING".
    
    Tại sao cần? Vì Anthropic API trả về 400 error nếu conversation
    history malformed. Synthetic results là "lie" nhỏ nhất để keep API happy.
```

---

## 7. ToolRouter — Tay Của Agent

### File: `agent/core/tools.py`

```text
ToolRouter
├── tools: dict[str, ToolSpec]     ← registry của built-in tools
├── mcp_client: Client | None      ← FastMCP client cho MCP servers
└── _mcp_initialized: bool         ← lazy init flag

ToolSpec (dataclass)
├── name: str
├── description: str               ← LLM đọc description này để biết dùng tool gì
├── parameters: dict               ← JSON Schema, LLM tạo args theo schema này
└── handler: Callable              ← async fn(args) → tuple[str, bool]
                                     str = result text
                                     bool = success flag
```

#### 7.1 Built-in Tools (16 tools)

```text
Research & Knowledge:
├── research          ── multi-step sub-agent với dedicated LLM calls
├── web_search        ── Tavily search
├── hf_papers         ── ArXiv papers từ HF daily papers
├── explore_hf_docs   ── search trong HF documentation tree
└── fetch_hf_docs     ── fetch specific doc page

HF Hub:
├── hf_inspect_dataset ── xem dataset structure, splits, features
├── hf_repo_files      ── đọc/ghi/xóa files trong HF repos
└── hf_repo_git        ── git operations (commit, push, history)

Compute:
├── sandbox_*          ── remote Python execution trong HF Space
│   ├── sandbox_create
│   ├── sandbox_exec
│   ├── sandbox_read_file
│   ├── sandbox_write_file
│   └── sandbox_status
└── hf_jobs            ── submit/monitor HF Training Jobs (GPU clusters)

GitHub:
├── github_find_examples ── search code examples
├── github_read_file     ── đọc file từ GitHub
└── github_list_repos    ── list repos

Utility:
├── plan               ── structured planning với step tracking
└── notify             ── gửi notification ra Slack/gateway
```

#### 7.2 MCP Integration

```text
ToolRouter.__init__():
    1. register tất cả built-in tools
    2. if mcp_servers config tồn tại:
           inject HF token vào headers của mỗi server
           tạo FastMCP Client với multi-server config

ToolRouter.__aenter__():  ← context manager (async with tool_router:)
    1. mcp_client.initialize()
    2. fetch tool specs từ MCP servers
    3. register MCP tools vào tools dict
    
execute_tool(name, args):
    if name in self.tools:
        return await self.tools[name].handler(args)
    elif self.mcp_client and name in mcp_tools:
        result = await mcp_client.call_tool(name, args)
        return convert_mcp_content_to_string(result), True
```

**Tại sao MCP cho HF Hub?** MCP (Model Context Protocol) là chuẩn mở cho tool calls. `hf-mcp-server` tại `huggingface.co/mcp?login` expose toàn bộ HF Hub API. Thay vì implement từng API call thủ công, agent dùng MCP để tự động có quyền truy cập vào mọi capability của HF Hub — kể cả các feature mới được thêm vào Hub sau này.

---

## 8. DoomLoop Detector — Hệ Thống Phòng Vệ

### File: `agent/core/doom_loop.py`

Đây là một trong những component thú vị nhất về mặt engineering. LLM agents có xu hướng bị stuck trong các vòng lặp vô tận — gọi cùng một tool với cùng arguments, không nhận ra mình đang lặp.

#### 8.1 Cấu Trúc Dữ Liệu

```python
@dataclass(frozen=True)
class ToolCallSignature:
    name: str         # tên tool
    args_hash: str    # MD5(canonical_json(args))[:12]
    result_hash: str  # MD5(str(result))[:12] — QUAN TRỌNG
```

**Tại sao hash cả result?** Nếu agent đang polling một job (gọi `hf_jobs` mỗi 30s), args giống nhau nhưng result khác (job status thay đổi). Đây là legitimate polling, không phải doom loop. Chỉ hash args sẽ false-positive cho polling. Hash cả result = chỉ trigger khi **cả args lẫn result đều giống hệt nhau**.

#### 8.2 Canonical JSON Normalization

```python
def _normalize_args(args_str: str) -> str:
    # LLM có thể gen: {"a": 1, "b": 2} hoặc {"b": 2, "a": 1}
    # Cả hai đều là cùng một call nhưng hash khác nhau nếu không normalize
    return json.dumps(json.loads(args_str), sort_keys=True, separators=(",", ":"))
```

#### 8.3 Hai Pattern Được Detect

```text
Pattern 1: Identical Consecutive (threshold=3)
─────────────────────────────────────────────
signatures = [A, B, C, C, C]  ← 3 C liên tiếp → DOOM LOOP!

Pattern 2: Repeating Sequence (length 2-5, reps ≥ 2)
──────────────────────────────────────────────────────
signatures = [X, Y, Z, A, B, A, B]  ← [A,B] lặp 2 lần → DOOM LOOP!
signatures = [X, A, B, C, A, B, C]  ← [A,B,C] lặp 2 lần → DOOM LOOP!
```

#### 8.4 Response Khi Detect

```text
Thay vì crash hoặc stop agent, inject một "system message" vào đầu messages:

"[SYSTEM: REPETITION GUARD] You have called 'sandbox_exec' with the same 
arguments multiple times in a row, getting the same result each time. 
STOP repeating this approach — it is not working. Step back and try a 
fundamentally different strategy..."

Tại sao inject vào messages thay vì throw exception?
LLM cần đọc được lý do tại sao nó bị dừng. Exception không giải thích được.
Message injection = LLM có thể self-correct và thử cách khác.
```

---

## 9. Backend — API Gateway & Session Pool

### Files: `backend/session_manager.py`, `backend/routes/agent.py`

#### 9.1 Session Pool Architecture

```text
SessionManager (singleton)
│
├── sessions: dict[str, AgentSession]
│   ├── session_id_1 → AgentSession { session, tool_router, queues, task, broadcaster }
│   ├── session_id_2 → AgentSession { ... }
│   └── ...
│
├── _lock: asyncio.Lock   ← guard create/delete operations
├── MAX_SESSIONS: 200     ← global cap
└── MAX_PER_USER: 10      ← per-user cap

Sizing rationale:
   8 vCPU / 32 GB RAM (HF Space tier)
   Mỗi session dùng ~10-20 MB (context, queues, asyncio task)
   200 sessions × 20 MB = 4 GB worst case
   Còn 28 GB cho Python runtime + per-request overhead
```

#### 9.2 `AgentSession` — Session Wrapper

```text
AgentSession (dataclass)
├── session_id: str
├── session: Session             ← agent state
├── tool_router: ToolRouter      ← tool registry
├── submission_queue: Queue      ← input channel
├── user_id: str                 ← owner (authorization)
├── hf_token: str | None         ← OAuth token của user
├── task: asyncio.Task           ← coroutine chạy agent loop
├── broadcaster: EventBroadcaster ← fan-out events đến SSE subscribers
├── is_active: bool
├── is_processing: bool          ← đang xử lý request?
└── claude_counted: bool         ← đã tính quota Claude chưa?
```

#### 9.3 EventBroadcaster — Fan-out Pattern

```text
EventBroadcaster
├── _source: asyncio.Queue      ← đọc từ agent's event_queue
└── _subscribers: dict[id, Queue]  ← mỗi SSE connection là một subscriber

run():
    while True:
        event = await _source.get()   ← 1 event từ agent
        for sub_q in _subscribers:
            await sub_q.put(event)   ← fan-out đến mọi subscriber
```

**Tại sao cần fan-out?** Một session có thể có nhiều SSE connections đồng thời (ví dụ: user mở cùng session trên 2 tab). EventBroadcaster đảm bảo mọi subscriber đều nhận được mọi event. Events đến khi không có subscriber nào sẽ bị discard — không buffer vì mỗi SSE turn là một request riêng biệt.

#### 9.4 Session Creation — Thread Pool cho Blocking I/O

```python
def _create_session_sync():
    # Blocking operations:
    # - ToolRouter.__init__: có thể call HF API
    # - Session.__init__: litellm.get_model_info() (HTTP call)
    # - ContextManager.__init__: load system prompt, whoami API
    tool_router = ToolRouter(config.mcpServers, hf_token=hf_token)
    session = Session(event_queue, config=session_config, ...)
    return tool_router, session

# Chạy trong thread pool để không block event loop
tool_router, session = await asyncio.to_thread(_create_session_sync)
```

**Tại sao quan trọng?** FastAPI chạy trên asyncio event loop. Nếu `Session.__init__` block event loop (vì HTTP call), toàn bộ server dừng xử lý requests trong thời gian đó. `asyncio.to_thread()` chạy blocking code trong thread pool, event loop tự do nhận requests khác.

#### 9.5 Session Rehydration — `seed_from_summary()`

```text
Vấn đề: User đóng tab (session vẫn sống trên server), sau đó mở lại.
Frontend có cached messages cũ. Server có session mới (context trống).

Giải pháp: seed_from_summary()
    1. Frontend gửi cached messages lên
    2. Backend gọi LLM để summarize chúng
    3. Inject summary vào context của session mới:
       "[SYSTEM: Your prior memory of this conversation — written 
        in your own voice right before restart. Continue from here.]"
    4. Agent "nhớ lại" context cũ mà không cần re-process toàn bộ history
```

**Tại sao không replay toàn bộ messages?** Nếu session cũ có 200 messages × 1000 tokens = 200k tokens, replay sẽ lập tức fill context window. Summarization giữ essence của conversation trong ~2000 tokens.

#### 9.6 Interrupt — Bypass Queue

```python
async def interrupt(session_id: str) -> bool:
    agent_session = self.sessions.get(session_id)
    agent_session.session.cancel()  # Set asyncio.Event trực tiếp
    return True
```

**Interrupt bypass queue**, không thêm vào submission_queue. Tại sao? Nếu queue đang có 5 operations đang chờ, thêm INTERRUPT vào cuối queue có nghĩa là agent phải xử lý xong 5 operations trước mới interrupt. Đó không phải interrupt, đó là "schedule interrupt later". Gọi `session.cancel()` trực tiếp = immediate signal.

---

## 10. Frontend — SSE Bridge & React Layer

### Files: `frontend/src/lib/sse-chat-transport.ts`, `frontend/src/hooks/useAgentChat.ts`

#### 10.1 Protocol Mismatch Problem

```text
Backend protocol:         Vercel AI SDK protocol:
─────────────────         ───────────────────────
data: {                   UIMessageChunk {
  event_type: "tool_call"   type: "tool-call"
  tool: "sandbox_exec"      toolCallId: "..."
  args: {...}               toolName: "sandbox_exec"
}                           args: {...}
                          }

data: {
  event_type: "assistant_chunk"  →  UIMessageChunk { type: "text-delta", textDelta: "..." }
  chunk: "Tôi sẽ..."
}
```

`SSEChatTransport` là adapter layer giải quyết mismatch này.

#### 10.2 SSEChatTransport — Dual Stream Architecture

```text
sendMessages() được gọi khi user submit message
        │
        ├── 1. POST /api/sessions/{id}/submit {text: "..."}
        │      → Backend enqueue USER_INPUT operation
        │
        ├── 2. fetch /api/sessions/{id}/events (SSE)
        │      → Nhận stream của AgentEvent objects
        │
        ├── 3. createSSEParserStream()
        │      TransformStream<string, AgentEvent>
        │      Parse "data: {...}\n\n" format
        │
        ├── 4. createEventToChunkStream(sideChannel)
        │      TransformStream<AgentEvent, UIMessageChunk>
        │      │
        │      ├── event_type == "assistant_chunk"
        │      │   → UIMessageChunk { type: "text-delta", textDelta }
        │      │
        │      ├── event_type == "tool_call"
        │      │   → UIMessageChunk { type: "tool-call-streaming-start" }
        │      │   → sideChannel.onToolCallPanel(tool, args)
        │      │
        │      ├── event_type == "tool_output"
        │      │   → UIMessageChunk { type: "tool-result" }
        │      │   → sideChannel.onToolOutputPanel(tool, output)
        │      │
        │      ├── event_type == "approval_required"
        │      │   → sideChannel.onApprovalRequired(tools)
        │      │   → UIMessageChunk approval request
        │      │
        │      ├── event_type == "turn_complete"
        │      │   → UIMessageChunk { type: "finish" }
        │      │   → sideChannel.onProcessingDone()
        │      │
        │      ├── event_type == "ready"
        │      │   → sideChannel.onReady()
        │      │
        │      └── event_type == "error"/"shutdown"/"interrupted"
        │          → sideChannel callbacks
        │
        └── 5. Return ReadableStream<UIMessageChunk> cho useChat
```

**Tại sao không dùng WebSocket?** SSE là uni-directional (server → client), đơn giản hơn WebSocket. Backend Agent emits events → client consumes. Khi user cần gửi message, dùng một POST request riêng. Hai operations riêng biệt (gửi + nhận) dễ reason hơn một bidirectional socket.

#### 10.3 State Management — Zustand Stores

```text
3 stores độc lập, có quan hệ:

agentStore (global state + per-session state)
├── connected: bool               ← session có đang kết nối không
├── isProcessing: bool            ← active session đang xử lý không
├── error: string | null
├── plan: PlanStep[] | null       ← current plan từ plan_tool
├── sessions: Map<id, SessionState>  ← per-session state
└── updateSession(id, partial)    ← update session, mirror to globals nếu active

sessionStore (session list)
├── sessions: SessionInfo[]       ← list của sessions
├── activeSessionId: string | null
└── setSessionActive(id, bool)

layoutStore (UI state)
├── isRightPanelOpen: bool        ← code/plan panel
└── setRightPanelOpen(bool)
```

**Tại sao tách 3 stores?** Single global store sẽ cause mọi component re-render khi bất kỳ state nào thay đổi. Tách stores theo domain = chỉ components cần `layoutStore` re-render khi panel mở/đóng, không ảnh hưởng đến Chat components.

#### 10.4 Multi-Session Architecture

```text
Frontend support nhiều sessions (như browser tabs):

SessionSidebar:
  [session_1] ── active ──► agentChat_1 mounted, useAgentChat running
  [session_2] ── inactive ► agentChat_2 mounted, useAgentChat suspended
  [+ New]

Mỗi session có own useAgentChat instance với own:
  - SSEChatTransport
  - Message store (localStorage key: `chat_messages_{sessionId}`)
  - Research state store
  - Backend message cache

Khi switch session:
  - isActive prop thay đổi
  - Side-channel callbacks check isActiveRef trước khi update global state
  - Chỉ active session mirror state lên global agentStore
```

---

## 11. Data Flywheel — Vòng Lặp Thu Thập Dữ Liệu

### Files: `agent/core/session.py`, `agent/sft/tagger.py`, `scripts/build_sft.py`

#### 11.1 Tại Sao "Data Flywheel"?

```text
Agent hoạt động tốt
    │
    ▼
Users dùng nhiều hơn
    │
    ▼
Thu thập nhiều session trajectories hơn
    │
    ▼
Train model tốt hơn trên trajectories đó
    │
    ▼
Agent hoạt động tốt hơn ──────────────────┐
    │                                      │
    └──────────────────────────────────────┘
                Flywheel!
```

#### 11.2 Session Trajectory Structure

```json
{
  "session_id": "uuid",
  "user_id": "hf_username",
  "session_start_time": "2026-04-28T10:00:00",
  "model_name": "anthropic/claude-opus-4-6",
  "total_cost_usd": 0.42,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Fine-tune llama on my dataset"},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    ...
  ],
  "events": [
    {"timestamp": "...", "event_type": "tool_call", "data": {...}},
    {"timestamp": "...", "event_type": "llm_call", "data": {"cost_usd": 0.05}},
    ...
  ],
  "tools": [...],
  "upload_status": "success"
}
```

#### 11.3 SFT Tagging System

```text
tag_session(trajectory) → list[str]

Tags được tạo tự động từ trajectory:

tool:<name>        → "tool:hf_jobs", "tool:sandbox_exec"
outcome:<end>      → "outcome:completed", "outcome:doom_loop"
hf_job:<facet>    → "hf_job:succeeded", "hf_job:oom"
gpu:<kind>         → "gpu:a100", "gpu:h100"
sandbox:<facet>    → "sandbox:created", "sandbox:long_lived"
model:<family>     → "model:opus", "model:kimi"
turns:<bucket>     → "turns:short" (<5), "turns:medium", "turns:long" (>20)
cost:<bucket>      → "cost:low" (<$0.10), "cost:med", "cost:high"
task:<kind>        → "task:training", "task:inference", "task:research_only"
```

Tags cho phép filter dataset downstream:
```python
# Chỉ lấy sessions training thành công với GPU
good_sessions = df[
    df.tags.apply(lambda t: 
        "outcome:completed" in t and 
        "task:training" in t and 
        "hf_job:succeeded" in t
    )
]
```

#### 11.4 Heartbeat Saves

```text
HeartbeatSaver.maybe_fire(session):
    Gọi sau mỗi send_event()
    
    Nếu elapsed > HEARTBEAT_INTERVAL:
        save_trajectory_local() ← không phải upload, chỉ local
        cập nhật _last_heartbeat_ts
    
    Mục đích: Nếu server crash giữa một long-running task
    (fine-tuning có thể mất nhiều giờ), heartbeat đảm bảo
    ta không mất toàn bộ trajectory. Upload sẽ retry sau.
```

---

## 12. Security Layer — Redact, Auth, Quotas

#### 12.1 Secret Redaction (`agent/core/redact.py`)

```text
scrub(trajectory_payload):
    Đệ quy qua dict/list/str
    Apply regex patterns:
    ├── hf_[A-Za-z0-9]{30,}    → [REDACTED_HF_TOKEN]
    ├── sk-ant-[...]{20,}      → [REDACTED_ANTHROPIC_KEY]
    ├── sk-[...]{40,}          → [REDACTED_OPENAI_KEY]
    ├── gh[pousr]_[...]{36,}   → [REDACTED_GITHUB_TOKEN]
    ├── github_pat_[...]{36,}  → [REDACTED_GITHUB_TOKEN]
    ├── AKIA/ASIA[A-Z0-9]{16}  → [REDACTED_AWS_KEY_ID]
    ├── Bearer <token>         → Bearer [REDACTED]
    └── KEY=value, KEY: value  → KEY=[REDACTED]

Redact xảy ra TẠI THỜI ĐIỂM SAVE (không phải trước khi agent xử lý).
Agent cần secrets để hoạt động. Secrets chỉ cần được xóa trước khi
lưu xuống disk hoặc upload.
```

#### 12.2 HF OAuth Auth (`backend/routes/auth.py`)

```text
GET /api/auth/user
    ← HF OAuth token từ cookie/header
    → gọi HF API /api/whoami-v2
    → trả về {username, isPro, orgs}

Tại sao không JWT? HF đã có identity system. Tái dùng HF token
= users không cần tạo account riêng cho ML Intern.
```

#### 12.3 Quota System (`backend/user_quotas.py`)

```text
Claude quota (Anthropic models):
├── Mỗi user có daily_claude_sessions_cap
├── Tính tại message-submit time (không phải session-create time)
│   Lý do: user có thể tạo Claude session để "nhìn around" mà không 
│   tốn quota. Chỉ tính khi thực sự dùng.
├── Flag claude_counted trên AgentSession để tránh double-count
│   (user switch model trong session thì count không đổi)
└── 429 Too Many Requests nếu vượt cap

Anthropic model gate:
├── Chỉ HF staff (in HuggingFace org) mới dùng được Claude Opus
├── Claude dùng ANTHROPIC_API_KEY của Space (bill cho HF)
├── Model free (Kimi, MiniMax, GLM) dùng HF Router (bill qua X-HF-Bill-To)
└── Non-HF users vẫn có thể dùng free models
```

#### 12.4 Prompt Caching — Cost Optimization

```text
with_prompt_caching(messages, tools, model_name):
    Chỉ áp dụng cho Anthropic models

    1. Tools block: thêm cache_control vào LAST tool spec
       → Toàn bộ tool definitions (~3-4k tokens) được cache
       → Các turns tiếp theo trong 5 phút: ~10% của input price

    2. System message: wrap content thành cached block
       → System prompt (~1-2k tokens) được cache

    Kết quả: mỗi turn chỉ phải trả full price cho NEW tokens
    (user message + tool results). Static context (tools + system) = free.
    
    Ví dụ: 20 turns × 5k tokens static = 100k tokens tiết kiệm mỗi session
    Với Claude Opus 4.6: ~$1.50 tiết kiệm mỗi session dài.
```

---

## 13. Các Quyết Định Thiết Kế Quan Trọng

### 13.1 LiteLLM — Model Abstraction Layer

```text
Vấn đề: Agent muốn support Anthropic, OpenAI, HF Router, Bedrock, ...
         Mỗi provider có API format khác nhau.

Giải pháp: litellm.acompletion() — unified interface
    litellm.acompletion(
        model="anthropic/claude-opus-4-6",  # hoặc "openai/gpt-5.5"
        messages=[...],                      # OpenAI format, litellm translate
        tools=[...],
        stream=True,
    )

Bonus: litellm.drop_params = True
    → Nếu một provider không support một param (ví dụ: thinking_effort),
      litellm tự drop param đó thay vì throw error.
      Agent không cần biết capabilities của từng provider.
```

### 13.2 LiteLLM Effort Cascade — Graceful Degradation

```text
Vấn đề: Mỗi model support các effort levels khác nhau.
         Claude Opus 4.7: "xhigh", "high", "medium", "low"
         Claude Opus 4.6: "max", "high", "medium", "low"  
         Kimi K2.6: không support thinking params

effort_probe.py:
    Thử: "max" → nếu 400 error:
    Thử: "high" → nếu 400 error:
    Thử: "medium" → nếu fail:
    Kết luận: model không support thinking, gửi None

    Kết quả được cache trong session.model_effective_effort[model_name]
    để lần sau không cần probe lại.
```

### 13.3 Local Mode

```text
Config option: local_mode = True

Trong ToolRouter, khi local_mode=True:
    → Không tạo sandbox (không cần HF Space)
    → Không expose hf_jobs tool
    → Dùng local_tools.py thay vì remote tools

Tại sao? CLI users chạy locally có thể không có HF token.
local_mode cho phép agent hoạt động với quyền giới hạn hơn.
```

### 13.4 YOLO Mode

```text
Config: yolo_mode = True / False (default: False)

_needs_approval():
    if session.config.yolo_mode:
        return False  # skip tất cả approvals

Normally cần approval:
    - hf_jobs: submit training job (tốn tiền)
    - sandbox: create/destroy Space
    - hf_repo_files: upload/delete files
    - hf_repo_git: destructive git ops

YOLO mode = agent tự approve mọi thứ. Hữu ích cho headless/automated runs.
```

---

## 14. Thứ Tự Đọc Codebase

Đọc theo thứ tự này để xây dựng mental model từ dưới lên trên.

### Phase 1 — Contracts & Config (30 phút)

| File | Mục tiêu |
|------|---------|
| [agent/config.py](agent/config.py) | Hiểu shape của Config — model, tools, messaging |
| [agent/tools/types.py](agent/tools/types.py) | ToolSpec dataclass |
| [agent/core/session.py](agent/core/session.py) | `OpType`, `Event`, `Session` — unit of conversation |
| [configs/cli_agent_config.json](configs/cli_agent_config.json) | Config thực tế |

### Phase 2 — Agent Heart (1 giờ)

| File | Mục tiêu |
|------|---------|
| [agent/core/agent_loop.py](agent/core/agent_loop.py) | `submission_loop` + `run_agent` — ĐỌC KỸ |
| [agent/core/tools.py](agent/core/tools.py) | `ToolSpec`, `ToolRouter`, MCP integration |
| [agent/context_manager/manager.py](agent/context_manager/manager.py) | Compaction, history management |
| [agent/core/doom_loop.py](agent/core/doom_loop.py) | Loop detection — ngắn nhưng elegant |
| [agent/core/prompt_caching.py](agent/core/prompt_caching.py) | Cache breakpoints cho Anthropic |

### Phase 3 — Tools (45 phút)

| File | Mục tiêu |
|------|---------|
| [agent/tools/sandbox_tool.py](agent/tools/sandbox_tool.py) | Remote code execution — tool quan trọng nhất |
| [agent/tools/research_tool.py](agent/tools/research_tool.py) | Sub-agent pattern |
| [agent/tools/jobs_tool.py](agent/tools/jobs_tool.py) | HF Training Jobs |
| Còn lại trong [agent/tools/](agent/tools/) | Scan nhanh, tất cả follow ToolSpec pattern |

### Phase 4 — CLI Entrypoint (20 phút)

| File | Mục tiêu |
|------|---------|
| [agent/main.py](agent/main.py) | Cách CLI wire queues, headless vs interactive |

### Phase 5 — Web Backend (45 phút)

| File | Mục tiêu |
|------|---------|
| [backend/session_manager.py](backend/session_manager.py) | Multi-tenant pool, EventBroadcaster |
| [backend/routes/agent.py](backend/routes/agent.py) | REST + SSE endpoints, quota |
| [backend/routes/auth.py](backend/routes/auth.py) | HF OAuth |
| [backend/dependencies.py](backend/dependencies.py) | Auth middleware |
| [backend/user_quotas.py](backend/user_quotas.py) | Daily caps |

### Phase 6 — Frontend (1 giờ)

| File | Mục tiêu |
|------|---------|
| [frontend/src/lib/sse-chat-transport.ts](frontend/src/lib/sse-chat-transport.ts) | **Critical** — SSE → AI SDK bridge |
| [frontend/src/hooks/useAgentChat.ts](frontend/src/hooks/useAgentChat.ts) | React hook wire mọi thứ |
| [frontend/src/store/agentStore.ts](frontend/src/store/agentStore.ts) | Zustand global state |
| [frontend/src/components/SessionChat.tsx](frontend/src/components/SessionChat.tsx) | Main chat component |
| [frontend/src/components/Chat/](frontend/src/components/Chat/) | Scan nhanh rendering components |

### Phase 7 — Data Flywheel (20 phút)

| File | Mục tiêu |
|------|---------|
| [agent/core/redact.py](agent/core/redact.py) | Secret scrubbing patterns |
| [agent/core/telemetry.py](agent/core/telemetry.py) | Cost tracking, heartbeat |
| [agent/sft/tagger.py](agent/sft/tagger.py) | Automatic session tagging |
| [scripts/build_sft.py](scripts/build_sft.py) | Trajectory → training data |

---

## Tóm Tắt Mental Model

Hệ thống này có thể được hiểu qua một analogy đơn giản:

```text
Hãy nghĩ về nó như một UNIX pipeline:

[User Input]         ←→  stdin
[submission_queue]   ←→  pipe (buffered, typed)
[agent_loop]         ←→  process (stateful, iterative)
[event_queue]        ←→  pipe (event stream)
[CLI/SSE/Web]        ←→  stdout (multiple sinks)
[session_logs]       ←→  tee (fork to log file)
[HF Dataset]         ←→  log aggregator

Và mỗi "pipe" là typed AsyncQueue — không phải byte stream,
mà là structured objects với clear contracts.
```

**Ba invariant quan trọng nhất của hệ thống:**

1. **Agent Core không biết về transport**: `agent_loop.py` không import FastAPI, không biết về SSE, không biết về CLI. Nó chỉ đọc từ `submission_queue` và write vào `event_queue`.

2. **Session là đơn vị cô lập**: Hai sessions không share state. Mỗi session có context, tools, config riêng. Session isolation = multi-tenancy là free.

3. **Every event is logged**: `session.send_event()` là single point of truth. Telemetry, data collection, notifications đều đi qua đây — không có out-of-band paths bị bỏ sót.
