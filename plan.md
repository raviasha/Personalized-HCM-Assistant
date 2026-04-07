# Personalized HCM Benefits Chatbot — Requirements & Plan

## 1. Overview

A prototype chatbot embedded within an HCM application that provides personalized benefits guidance to employees. The system uses RAG (Retrieval-Augmented Generation) over a benefits knowledge corpus stored in a vector database, combined with per-employee persistent memory (JSON state) that evolves across conversations.

**Stack:** Python + Gradio | OpenAI GPT-4o + text-embedding-3-small | ChromaDB (embedded) | Render free tier

---

## 2. Functional Requirements

### FR-1: Benefits Knowledge Corpus
- Curate a synthetic but realistic corpus of employee benefits documents covering:
  - Health insurance (medical, dental, vision) — plan tiers, enrollment windows, costs
  - Retirement plans (401k match, vesting schedules, contribution limits)
  - PTO / leave policies (vacation accrual, sick leave, parental leave, FMLA)
  - Life & disability insurance (coverage levels, beneficiary rules)
  - Wellness programs (gym reimbursement, EAP, mental health)
  - FSA / HSA accounts (contribution limits, eligible expenses, rollover rules)
  - Tuition reimbursement / professional development (eligibility, caps, process)
- Data sourced from: synthetic company handbook + factual IRS/DOL reference data (Publication 969, 401k limits, FMLA rules)
- Documents chunked, embedded with `text-embedding-3-small`, stored in ChromaDB
- ~50-100 document chunks total

### FR-2: Employee State (Long-Term Memory)
- Each employee has a JSON profile capturing:
  - **Static attributes:** name, employee_id, department, hire_date, age, family_status (single/married/dependents)
  - **Enrolled benefits:** current plan selections (health tier, 401k contribution %, FSA/HSA elections)
  - **Interests:** topics they've asked about (array, max 10 items, evolves via add/remove)
  - **Unresolved questions:** open questions not yet answered (array, max 10, removed when resolved or auto-expired after 3 sessions without mention)
  - **Pending actions:** things the employee said they'd do or want to explore (array, max 5, removed when completed/abandoned or auto-expired after 3 sessions)
  - **Interaction history:** summary of past conversations (array, max 5 recent verbatim; older entries collapsed into `historical_summary` string via GPT-4o)
  - **Historical summary:** single string summarizing all compacted old interaction history entries
  - **Seen recommendations:** list of topic IDs already surfaced (no compaction, low cost)
  - **Last interaction timestamp**
  - **Session counter:** incremented each login, used for expiry calculations
- State loaded on login, updated after each conversation via bidirectional delta (add + remove)

### FR-3: Synthetic Employees (3 profiles)
1. **Maria Chen** — Age 28, single, software engineer, newly hired (3 months). Enrolled in basic health plan only. Has not explored retirement or HSA options. Interest: understanding 401k match.
2. **James Williams** — Age 42, married with 2 kids, operations manager, 8 years tenure. Enrolled in family health plan + 401k (6%). Recently asked about FSA vs HSA. Interest: optimizing tax-advantaged accounts.
3. **Priya Sharma** — Age 55, married, VP of Finance, 15 years tenure. Max 401k contribution, premium health plan. Approaching retirement. Interest: catch-up contributions, Medicare transition, succession planning leave.

### FR-4: Proactive Recommendation Engine
- On login, the system:
  1. Loads employee state JSON
  2. Queries ChromaDB for documents relevant to the employee's profile (age, family status, enrolled plans, interests) BUT not already in `seen_recommendations`
  3. Ranks results by relevance to employee context
  4. Opens conversation with 1-2 personalized recommendations (e.g., "Hi Maria! Since you're new and haven't set up your 401k yet, did you know the company matches up to 4%? I can help you understand your options.")

### FR-5: Conversational RAG
- User can ask free-form questions about benefits
- System retrieves relevant chunks from ChromaDB, augments prompt with employee state, generates response via GPT-4o
- Conversation maintains session context (multi-turn)
- System prompt includes employee profile for personalization

### FR-6: Post-Conversation Memory Update (Bidirectional)
- After conversation ends (user clicks "End Chat" or session timeout):
  1. GPT-4o analyzes the conversation transcript against the current employee state
  2. Produces a **structured delta with both ADDITIONS and DELETIONS**:
     - **Additions (add):** new interests discovered, new unresolved questions, new action items, new topics discussed
     - **Deletions (remove):** questions now resolved (remove from `unresolved_questions`), interests fully addressed / no longer relevant (remove from `interests`), action items completed or abandoned (remove from `pending_actions`), outdated concerns superseded by new info
  3. Delta format example:
     ```json
     {
       "add": {
         "interests": ["HSA tax advantages"],
         "pending_actions": ["Compare HSA vs FSA eligible expenses"],
         "interaction_summary": "Asked about HSA basics and eligible expenses"
       },
       "remove": {
         "unresolved_questions": ["What is the difference between FSA and HSA?"],
         "pending_actions": ["Look into tax-advantaged accounts"]
       },
       "update": {
         "last_interaction_timestamp": "2026-04-07T14:30:00Z"
       }
     }
     ```
  4. Merge engine applies delta: appends `add` items to arrays, removes matching `remove` items from arrays, overwrites `update` fields
  5. Updates `seen_recommendations` with any topics surfaced during conversation
- Guardrails:
  - LLM output validated against a strict JSON schema before merge
  - Deletions require exact match against existing state items (fuzzy match with threshold)
  - No enrollment status changes without explicit employee confirmation in transcript
  - No PII hallucinated into state

### FR-7: Login Simulation
- Gradio dropdown to select employee (simulates authentication)
- On selection, state loads and proactive message appears
- "End Chat" button triggers memory update

### FR-8: State Compaction (Anti-Bloat)
- **Token budget:** Employee state JSON injected into prompts must stay ≤1500 tokens
- **Compaction runs on login** (state load time), before the conversation starts — no mid-chat latency
- **Per-field compaction rules:**
  | Field | Cap | Eviction strategy |
  |-------|-----|-------------------|
  | `interaction_history` | 5 entries | When >5, GPT-4o summarizes oldest entries into `historical_summary` string, then deletes them from array |
  | `interests` | 10 items | When >10, drop oldest items (by session added) not referenced in last 3 sessions |
  | `unresolved_questions` | 10 items | Auto-expire items not referenced in 3+ sessions (assumed stale) |
  | `pending_actions` | 5 items | Auto-expire items not referenced in 3+ sessions (assumed abandoned) |
  | `seen_recommendations` | No cap | Negligible tokens (just IDs) |
- **Token gate:** After per-field compaction, measure total state token count. If still >1500, trigger aggressive summarization of `historical_summary` + trim longest array items until under budget.
- **Each array item carries metadata:** `{ "text": "...", "added_session": N, "last_referenced_session": N }` to enable expiry logic

---

## 3. Non-Functional Requirements

### NFR-1: Render Free Tier Compatibility
- Max 512MB RAM — ChromaDB embedded in-process, small corpus (~50-100 chunks)
- Ephemeral filesystem — pre-compute embeddings at build time; store corpus + embeddings as files in repo; rebuild ChromaDB collection on startup
- Employee state: for prototype, store JSON files in repo as defaults; in-session modifications held in memory; option to persist to Render Postgres (free, 1GB, 30-day expiry)
- Cold start ~30-60s acceptable (spin-down after 15min inactivity)

### NFR-2: Cost
- OpenAI API costs minimal: ~$0.02/1M tokens for embeddings, ~$2.50/1M tokens for GPT-4o input
- Prototype usage: <$1 total estimated

### NFR-3: Security
- OpenAI API key stored as Render environment variable, never in code
- No real PII — all synthetic data
- Input sanitization on user messages (prompt injection mitigation)

---

## 4. Architecture

```
┌─────────────────────────────────────────┐
│              Gradio UI                  │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │ Employee  │  │    Chat Interface    │ │
│  │ Selector  │  │  (multi-turn)       │ │
│  └──────────┘  └──────────────────────┘ │
└────────────────────┬────────────────────┘
                     │
        ┌────────────▼────────────┐
        │     App Controller      │
        │  (session management)   │
        └──┬─────────┬────────┬───┘
           │         │        │
   ┌───────▼──┐  ┌───▼────┐  ┌▼──────────┐
   │ ChromaDB │  │ OpenAI │  │ Employee  │
   │ (in-mem) │  │  API   │  │ State Mgr │
   │ Benefits │  │GPT-4o +│  │  (JSON)   │
   │ Corpus   │  │Embeddings│ └───────────┘
   └──────────┘  └────────┘
```

### Key Files
```
/
├── app.py                    # Gradio app, main entry point
├── requirements.txt          # Dependencies
├── corpus/
│   ├── health_insurance.md   # Benefits documents (markdown)
│   ├── retirement_401k.md
│   ├── pto_leave.md
│   ├── life_disability.md
│   ├── wellness.md
│   ├── fsa_hsa.md
│   └── tuition_reimbursement.md
├── embeddings/
│   └── corpus_embeddings.pkl # Pre-computed embeddings (built at startup or cached)
├── employees/
│   ├── maria_chen.json
│   ├── james_williams.json
│   └── priya_sharma.json
├── src/
│   ├── vector_store.py       # ChromaDB initialization, query, rebuild
│   ├── state_manager.py      # Load/save/merge employee state
│   ├── state_compactor.py    # Token budget enforcement, field-level compaction, expiry
│   ├── recommender.py        # Proactive recommendation logic
│   ├── rag.py                # RAG pipeline (retrieve + augment + generate)
│   ├── memory_updater.py     # Post-conversation state extraction (add/remove deltas)
│   └── prompts.py            # System prompts, templates
├── render.yaml               # Render deployment config
└── README.md
```

---

## 5. Implementation Steps (UI-First, Feature-by-Feature)

Each step is self-contained and testable. Clear context between steps.

### Step 1: Project Setup + Skeleton Gradio App
- Create directory structure, `requirements.txt`, `__init__.py` files
- Build `app.py` — skeleton Gradio UI with: employee dropdown (hardcoded names), chatbot panel, message input, "End Chat & Save" button, collapsible "What I Know About You" JSON panel
- **No backend logic** — just echo messages, placeholder state
- **Test:** App launches locally, UI renders, can type messages

### Step 2: Corpus + Vector Store + Basic RAG Chat
- Write 7 synthetic benefits documents in `corpus/` (~500-1000 words each)
- Build `src/vector_store.py` — load markdown, chunk, embed with OpenAI, store in ChromaDB in-memory
- Build `src/rag.py` + `src/prompts.py` — retrieve relevant chunks + generate answer via GPT-4o
- Wire into `app.py` — chat now answers benefits questions via RAG (no employee context yet)
- **Test:** Ask "What is the 401k match?" → get a real answer from corpus

### Step 3: Employee Profiles + State Manager + Login Selector
- Create 3 employee JSON profiles in `employees/` with full schema (interests, unresolved_questions, pending_actions, etc.)
- Build `src/state_manager.py` — load/save employee state (local JSON files first, Supabase later)
- Wire into `app.py` — selecting employee loads their state into the "What I Know About You" panel; RAG prompt now includes employee context
- **Test:** Select Maria → see her state JSON; ask a question → response is personalized to Maria

### Step 4: Proactive Recommender
- Build `src/recommender.py` — on employee select: query ChromaDB with profile context, filter out seen_recommendations, generate personalized greeting via GPT-4o
- Wire into `app.py` — on employee change, greeting message auto-appears in chat
- **Test:** Select Maria → see "Hi Maria, since you haven't set up your 401k yet..." greeting

### Step 5: Memory Updater (Bidirectional Delta)
- Build `src/memory_updater.py` — on "End Chat": analyze transcript vs current state, produce add/remove/update delta, validate against schema, merge into state, save
- Wire into `app.py` — "End Chat & Save" triggers memory update, state panel refreshes
- **Test:** Chat with Maria about HSA → End Chat → verify state JSON now has HSA in interests + any resolved questions removed

### Step 6: State Compactor
- Build `src/state_compactor.py` — on login: run expiry checks, summarize old history entries, enforce 1500-token budget
- Wire into `app.py` — compaction runs after state load, before recommender
- **Test:** Manually bloat a state JSON → login → verify it gets compacted

### Step 7: Persistence + Deployment
- Add Supabase integration to `src/state_manager.py` (read/write employee state to Postgres)
- Create `render.yaml`, `.env.example`, seed script for Supabase
- Deploy to Render with env vars: `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`
- **Test:** Deploy on Render free tier, verify state persists across restarts

---

## 6. Verification

1. **Unit:** Each module testable independently — vector_store returns results, state_manager round-trips JSON, memory_updater produces valid deltas, state_compactor enforces caps
2. **Integration:** Full flow test: select Maria → get recommendation about 401k → ask about HSA → end chat → verify state updated with HSA interest + 401k question resolved
3. **Persona coverage:** Test all 3 employees get distinct, relevant recommendations
4. **Compaction:** Simulate 10+ sessions for one employee → verify interaction_history stays at 5 entries + historical_summary grows, stale questions auto-expire, total state stays ≤1500 tokens
5. **Bidirectional delta:** Verify both additions AND deletions occur: resolved questions removed, new interests added, abandoned actions cleaned up
6. **Edge cases:** Empty conversation (no state update), repeated login (no duplicate recommendations), long conversation (memory extraction handles large transcripts)
7. **Deployment:** Verify app loads on Render within 60s, survives spin-down/wake cycle, stays under 512MB RAM

---

## 7. Decisions & Scope

- **In scope:** Single-session prototype, synthetic data, 3 employees, all benefit types, Render deployment, state transparency panel, state compaction
- **Out of scope:** Real authentication, real benefits data, multi-user concurrency, admin interface
- **Employee state persistence:** Supabase Postgres (free tier) for durable state that survives Render restarts. Employee JSON stored as JSONB column. Supabase URL + key stored as Render env vars.
- **Corpus style:** Fictional company handbook ("Acme Corp") with realistic plan names, tiers, and numbers grounded in real IRS/DOL 2025-2026 figures.
- **UI transparency:** "What I Know About You" collapsible panel in Gradio showing the current employee state JSON (read-only), updated live after memory save.
- **Embedding strategy:** Rebuild ChromaDB from corpus files on every app startup (~5-10s for 100 chunks). Pre-computed embeddings cached in `embeddings/` to avoid re-calling OpenAI API on every restart.
- **State compaction:** Tiered hybrid — per-field caps with session-based expiry, history summarization, 1500-token budget gate. Runs on login.
