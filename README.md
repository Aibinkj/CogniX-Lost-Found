# Lost & Found AI Agent

An agentic prototype that reunites people with lost property. A user describes
what they lost in plain language; the agent extracts the details, searches what
has actually been handed in, decides how confident it is, demands proof of
ownership before releasing anything, and escalates to a human when it cannot be
sure.

```
"I lost my black Sony headphones near the library yesterday around 4 PM."

  -> extract: category=headphones, brand=Sony, colour=black, location=library, time=yesterday 16:00
  -> search_matches()          top similarity 0.76
  -> confidence                0.89 HIGH
  -> "Please describe one distinctive feature of your item."
  -> "There is a scratch on the left earcup."
  -> verify_ownership()        PASS
  -> create_pickup_request()   LF1024
  -> "Match verified. Your headphones are available at Library Security Desk."
```

---

## Problem statement

University and transit lost-property desks hold thousands of items in plain
tables: a category, a colour, a free-text note. The people looking for those
items describe them completely differently ("my headset", "black Sony cans",
"wireless over-ears"). Keyword search fails, so staff read the list out loud and
the claimant says "yes, that's mine."

That creates two failures at once. Genuine owners never find their property
because the words don't line up. And items get handed to the wrong person,
because *describing* an item is not the same as *owning* it — and once the desk
has read out "black Sony headphones", anyone can repeat it back.

## Solution

Three things, in order:

1. **Semantic retrieval.** Item descriptions are embedded and matched by
   meaning, so "my headset" finds "wireless over-ear headphones".
2. **Multi-signal confidence.** Similarity alone is not evidence — "black Sony
   headphones" and "black Bose headphones" embed almost identically. The agent
   blends similarity with location, time, category and brand into an explainable
   score, and refuses to act below a threshold.
3. **Ownership verification.** Every item carries `hidden_features` — marks only
   the owner would know. They are never searched, never displayed, and never
   sent to an LLM unless the user has already been matched to that one item. The
   claimant must produce one before anything is released.

## Why this is an AI agent, not a chatbot

A chatbot maps input to a reply. This system **decides what to do next** based on
state it has accumulated, and its decisions change the world:

| Agent property | Where it shows up |
| --- | --- |
| Perception | `classify_intent` decides what a message is; `understand_request` parses item reports into structured fields |
| Memory | State persists across turns; the verification handshake spans two messages |
| Tool use | Ten tools that read and write PostgreSQL — no simulated calls |
| Branching on state | Four routers pick the next node from intent, confidence, ambiguity and verification outcome |
| Goal-directed loops | Ambiguous matches trigger clarification, re-extraction and re-search |
| Knowing its limits | Below threshold, or on failed verification, it escalates instead of guessing |
| Observability | Every decision is traced and rendered in the UI |

The agent can end a turn in eight different states (ask for more information, ask
a clarifying question, request ownership proof, create a pickup, escalate, answer
a help question, report a reference's status, explain its scope) and nothing in
the code decides that in advance.

This is a **single agent with multiple tools**, by design — the briefing's bar is
"more than one agent, *or* an agent that calls real tools".

---

## Architecture

```
                       Streamlit UI                      FastAPI
                    (streamlit_app.py)                 (app/main.py)
                             \                             /
                              \___________________________/
                                          |
                                 LangGraph agent
                              (app/agent/graph.py)
                                          |
             +----------------+-----------+-----------+----------------+
             |                |                       |                |
        extraction        tools layer            confidence        escalation
    (LLM or rules)      (app/tools/*)        (app/safety/*)     (app/safety/*)
             |                |                       |                |
             |          RAG retrieval                 |                |
             |        (app/rag/*)                     |                |
             +----------------+-----------------------+----------------+
                                          |
                             PostgreSQL + pgvector
                            (app/database/*)
```

### Technology stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Orchestration | LangGraph | Explicit state machine with conditional edges |
| LLM | Configurable | OpenAI-compatible or Anthropic; optional |
| Embeddings | Sentence Transformers | `all-MiniLM-L6-v2`, 384 dims |
| Vector store | PostgreSQL + pgvector | Cosine distance, ivfflat index |
| ORM | SQLAlchemy 2.0 | Typed declarative models |
| API | FastAPI | Chat, search, verify, staff endpoints |
| UI | Streamlit | Chat plus an agent trace panel |
| Tracing | Langfuse | Optional; no-ops without credentials |
| Tests | pytest | 83 tests against a real database |

### Graceful degradation

Three dependencies are optional, and each has a working fallback rather than a
hard failure. The active choice is always reported in `/health`, the Streamlit
sidebar and the trace, so a fallback is never mistaken for the real thing.

| Missing | Fallback | Cost |
| --- | --- | --- |
| pgvector extension | `REAL[]` column, NumPy cosine ranking | No ANN index; fine at prototype scale |
| `sentence-transformers` | Deterministic hashed word/trigram embedder | Lexical, not semantic — weaker on synonyms |
| LLM API key | Rule-based keyword/regex extractor | No paraphrase handling; the demo still runs |

> **Note on this machine.** The prototype was developed and tested against a
> local PostgreSQL 16 that does not ship the `vector` extension, so it ran on
> the NumPy path. `docker compose up -d` starts a `pgvector/pgvector:pg16`
> instance and the pgvector path activates automatically — no code change.

---

## LangGraph workflow

```
                                START
                                  |
                           CLASSIFY_INTENT
                                  |
               +------------------+-----------------+------------+-------------+
               |                  |                 |            |             |
            answer             report            status        help          other
        (proof pending)           |                 |            |             |
               |                  |            LOOKUP_CASE  SEARCH_HELP  OUT_OF_SCOPE
               |                  |               (end)        (end)         (end)
       VERIFY_OWNERSHIP   UNDERSTAND_REQUEST
               |                  |
               |        NEED_MORE_INFORMATION?
               |          yes            no
               |           |              |
               |        ASK_USER    SEARCH_MATCHES
               |          (end)           |
               |                 CALCULATE_CONFIDENCE
               |                          |
               |                  CONFIDENCE_ROUTER
               |             LOW      MEDIUM/AMBIGUOUS     HIGH
               |              |             |               |
               |     HUMAN_ESCALATION  ASK_CLARIFICATION  REQUEST_VERIFICATION
               |              |            (end)            (end)
        VERIFICATION_ROUTER   |
         PASS        FAIL ----+
          |
      CREATE_PICKUP
          |
      NOTIFY_USER
          |
         END
```

A turn ends whenever the agent needs the user. `ASK_USER`,
`ASK_CLARIFICATION` and `REQUEST_VERIFICATION` set `awaiting` and stop; the next
message re-enters the graph with the same state. That is how the two-turn
verification handshake works without holding a thread open, and it keeps the
state serialisable so the API can hand it back to the client.

**Routers** (`app/agent/nodes.py`):

- `intent_router` — side questions (help, status, small talk) are answered without
  touching the case; otherwise resume a pending verification or run the item search
- `need_more_information` — refuse to search on a description too thin to rank
- `confidence_router` — LOW escalates, MEDIUM/ambiguous clarifies, HIGH verifies
- `verification_router` — PASS creates a pickup, FAIL escalates

Clarification is capped at `MAX_CLARIFICATION_ROUNDS` (2); past that the agent
escalates rather than badgering the user in a loop.

## Intent routing and help questions

Not every message is a lost-item report. "How can I reach this staff?" used to be
parsed as an item description and answered with "what's the category?". Every
message now passes through `classify_intent` first (`app/agent/intent.py`):

| Intent | Example | Handled by |
| --- | --- | --- |
| `report` | "I lost my black Sony headphones…" | the item search below |
| `answer` | "There's a scratch on the left earcup." | `verify_ownership`, only while proof is pending |
| `help` | "When is the library desk open?" | `search_help()` over the help knowledge base |
| `status` | "Any update on LF1024?" | `lookup_case()` |
| `other` | "hi", "what can you do?" | a scope explanation — no search |

- **Rules first, LLM second.** Deterministic patterns decide every clear case. The
  LLM is asked only about messages the rules can't place.
- **Verification routing never depends on a model.** While an ownership answer is
  pending, only the rules run. A side question there is answered and the check
  stays open — it is not counted as a failed attempt.
- **Grounded or silent.** Help answers come only from `help_articles` retrieved at
  or above `HELP_MIN_SIMILARITY` (0.30). Below that the agent says it doesn't know
  and gives the staff contact. An LLM answer that cites no source is discarded
  for the retrieved article itself. Every answer shows its `[n]` sources.
- **Context-aware.** Mid-verification, "this staff" resolves to the desk holding
  the matched item. After an escalation, the reply quotes the `ESC` reference.
  Escalation messages now include how to reach staff.

> **Sample data.** The desk hours, extensions, emails and policies in
> `scripts/seed_database.py` are placeholders, stored with `is_sample = TRUE` and
> labelled "Sample data" in the UI. Replace them with real details before real use.
> Re-running the seed script (with or without `--keep`) syncs them by slug.

## RAG architecture

```
query text -> embedding (384-d, L2-normalised) -> pgvector `<=>` cosine -> top-K
```

`app/rag/embeddings.py` builds the vectors; `app/rag/vector_search.py` ranks
them. Retrieval builds results from `FoundItem.public_dict()`, so
`hidden_features` **cannot** leak through the search path — that is a structural
guarantee, not a filter someone has to remember to apply. There is a test for it.

`app/tools/search.py` is the tool-layer entry point. Its optional `category`
argument post-filters a *wider* pull rather than constraining the search, so a
mis-extracted category can never silently hide the right item.

## Database schema

Reference DDL in [`app/database/schema.sql`](app/database/schema.sql); the ORM in
`app/database/models.py` is the source of truth.

| Table | Purpose |
| --- | --- |
| `users` | Reporters. No authentication in this prototype. |
| `found_items` | Handed-in items, embeddings, and `hidden_features` |
| `lost_items` | User reports, embedded for reverse matching |
| `claims` | One claim attempt against one item, with its verification outcome |
| `pickup_requests` | `LF####` reservations for verified claims |
| `escalations` | Staff review cases with the scores behind the decision |
| `notifications` | Messages sent to users |
| `help_articles` | Desk directory and FAQ entries, embedded for help questions |

`found_items` in full:

```
id  category  brand  color  description  hidden_features  location  found_time  status  embedding
```

`hidden_features` holds semicolon-separated marks, e.g.
`"scratch on left earcup; small sticker inside case"`.

## Tool architecture

Plain Python functions with typed signatures that read and write the real
database — no hard-coded responses. The LangGraph nodes call them directly, and
an MCP server or an LLM tool-calling loop can wrap the same functions unchanged.

| Tool | Module | Effect |
| --- | --- | --- |
| `report_lost_item()` | `tools/lost_item.py` | Persists and embeds the report |
| `register_found_item()` | `tools/found_item.py` | Adds and indexes a handed-in item |
| `search_matches()` | `tools/search.py` | Top-K semantic retrieval |
| `verify_ownership()` | `tools/verification.py` | Scores the answer, records a claim |
| `create_pickup_request()` | `tools/pickup.py` | Reserves the item, marks it claimed |
| `notify_user()` | `tools/notification.py` | Persists a message |
| `escalate_to_staff()` | `safety/escalation.py` | Opens a review case |
| `search_help()` | `tools/help.py` | Top-K retrieval over desks and FAQs |
| `get_desk_contact()` | `tools/help.py` | Contact details for the desk holding an item |
| `lookup_case()` | `tools/case_status.py` | Status of an `LF` pickup or `ESC` case, no case contents |

MCP was deliberately left out of the core path. The functions are already
MCP-shaped (typed arguments, JSON-serialisable returns), so a FastMCP wrapper is
additive — it would not have made the prototype work better, only later.

## Confidence mechanism

Five signals, explicit weights, configurable in `.env`:

| Signal | Weight | Measures |
| --- | --- | --- |
| Semantic similarity | 50% | Cosine similarity of the embeddings |
| Location | 20% | Exact, synonym-group, or word-overlap match |
| Time proximity | 15% | Linear decay over 48h; hard penalty if handed in *before* the loss |
| Category | 10% | Exact, or a known-confusable neighbour (headphones/earbuds) |
| Brand | 5% | Exact match, or no brand recorded |

Two details that matter more than the weights:

- **Unknown signals are neutral, not zero.** If the user never said *where*, that
  weight is redistributed across the signals that *are* known. Otherwise a terse
  but perfectly accurate report would be punished for being terse.
- **Colour is a multiplier, not a weighted signal.** It is the strongest
  disambiguator between otherwise identical items, so a stated-and-contradicted
  colour applies a direct 0.80x penalty rather than nudging a 5% term.
- **A different category is penalised the same way (0.70x).** As a 10% weight
  alone, a wallet at the library outranked the only set of keys for "I lost my
  keys near the library".
- **So is a contradicted brand (0.80x).** At 5% alone, "green Samsung phone, at
  the bus stop" scored the green *Google* phone at the bus stop as HIGH.
- **The clarification cap counts only unhelpful replies.** A reply that adds a
  new fact (category, brand, colour, location or time) resets it, so a user who
  finally says "a green Samsung" is asked what's still missing, not escalated.
- **Thin reports are asked about, not escalated.** "I lost my keys" scores LOW
  because the user said little, not because nothing fits. When the report has no
  location or no colour/brand and a same-category item is held, the agent asks for
  the missing details (at most twice) before handing over to staff.

Output:

```json
{
  "score": 0.892,
  "level": "HIGH",
  "reasons": [
    "Strong semantic similarity (0.76)",
    "Location matches (Library Security Desk)",
    "Found within 1h of the reported loss",
    "Category matches (headphones)",
    "Brand matches (Sony)",
    "Colour matches (black)"
  ],
  "signals": {"semantic": 0.759, "location": 0.9, "time": 0.98, "category": 1.0, "brand": 1.0, "color": 1.0},
  "weights": {"semantic": 0.5, "location": 0.2, "time": 0.15, "category": 0.1, "brand": 0.05}
}
```

Levels: `HIGH >= 0.75` verify · `0.60 <= MEDIUM < 0.75` clarify · `LOW < 0.60` escalate.

## Ownership verification

The system never reveals everything it knows about an item before ownership is
proved. Verification is deterministic by default — no LLM required, and no
`hidden_features` in any prompt.

The claimant's answer is scored against each recorded mark by **weighted token
coverage**, with stem and synonym handling: *"the left ear cup is scuffed"*
matches *"scratch on left earcup"*. Low-information modifiers ("small", "inside")
carry 0.4 weight; identifying nouns carry 1.0. The best-matching mark wins, and
the score must clear `VERIFICATION_THRESHOLD` (0.55).

Three safeguards worth calling out:

- **Generic-answer guard.** An answer made only of publicly visible attributes —
  *"it's black"*, *"they're Sony"* — is rejected outright, because the search
  result already told the user those.
- **The question gives no examples.** Asking *"a sticker? an engraving?"* would
  narrow a guesser's search space, and a category-tailored hint leaks something
  about the matched item. The prompt is identical for every item.
- **The failure reason never quotes the evidence.** The claimant reads it.

```python
verify_ownership(candidate_id=1, user_answer="There is a scratch on the left earcup.")
# {"verified": true, "confidence": 1.0,
#  "reason": "User-provided distinctive feature matches stored ownership evidence.",
#  "method": "token-overlap"}
```

Setting `VERIFICATION_USE_LLM=true` adds an LLM judge as a *second* opinion —
only that one item's marks are sent, and both signals must accept.

## Human escalation

The agent escalates rather than making a claim when confidence is below
threshold, verification fails, or candidates stay ambiguous after clarification.
An escalation is a durable record — the request, the candidates considered, their
scores, and the verification outcome — so staff can review the decision:

```
"Your item could not be confidently verified. I've escalated this case to staff
 for review, and they'll follow up with you. Reference: ESC500."
```

A node that crashes also escalates. Failing safe means failing to a human, never
to a guess.

---

## Installation

Requires Python 3.11+ and a PostgreSQL instance.

```bash
git clone <repo> && cd lost-found-agent
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install sentence-transformers                     # optional, recommended
pip install langfuse                                  # optional
cp .env.example .env
```

### Database

**With Docker (recommended — this is the pgvector path):**

```bash
docker compose up -d
# then set in .env:
# DATABASE_URL=postgresql+psycopg2://lostfound:lostfound@localhost:5433/lostfound
```

Port 5433 is used on the host so it will not collide with a local PostgreSQL.

**With an existing PostgreSQL:**

```sql
CREATE DATABASE lostfound;
```

and point `DATABASE_URL` at it. If the `vector` extension is unavailable the app
logs a warning and uses the NumPy path.

## Environment variables

Full documented list in [`.env.example`](.env.example). The ones that matter:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | local postgres | SQLAlchemy connection string |
| `VECTOR_BACKEND` | `auto` | `auto`, `pgvector` (fail loudly), or `numpy` |
| `EMBEDDING_BACKEND` | `auto` | `auto`, `sentence-transformers`, or `hashing` |
| `LLM_PROVIDER` | `none` | `openai`, `anthropic`, or `none` |
| `LLM_MODEL` / `LLM_API_KEY` / `LLM_BASE_URL` | — | Any OpenAI-compatible endpoint |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.75` | At or above: request verification |
| `CONFIDENCE_LOW_THRESHOLD` | `0.60` | Below: escalate |
| `AMBIGUITY_MARGIN` | `0.07` | Top-two gap that counts as ambiguous |
| `WEIGHT_*` | see above | Confidence weights; normalised automatically |
| `VERIFICATION_THRESHOLD` | `0.55` | Coverage needed to pass |
| `LANGFUSE_*` | — | Leave blank to disable tracing |

API keys are read from the environment only. Nothing is hard-coded, and `.env`
is gitignored.

## How to run

```bash
docker compose up -d                 # 1. database
python scripts/seed_database.py      # 2. schema + 27 found items + help articles
streamlit run streamlit_app.py       # 3. UI at http://localhost:8501
```

The FastAPI backend is optional — Streamlit calls the agent in-process:

```bash
uvicorn app.main:app --reload        # http://localhost:8000/docs
curl http://localhost:8000/health
```

### How to seed the database

```bash
python scripts/seed_database.py          # drop and re-create everything
python scripts/seed_database.py --keep   # only add rows if the table is empty
```

27 found items across headphones, earbuds, phones, wallets, backpacks, laptops,
watches, water bottles, books, chargers, keys, glasses and umbrellas. The
catalogue deliberately includes near-misses — a black *Bose* headset and a *navy*
Sony headset alongside the black Sony pair — so similarity alone is not enough
and the confidence signals have to do real work.

Re-seed before a demo: a completed pickup marks its item `claimed`, which
correctly removes it from search.

### How to run tests

```bash
python -m pytest              # 83 tests
python -m pytest -v tests/test_verification.py
```

Tests create and seed a separate `<database>_test` database, so development data
is never touched. They run against real PostgreSQL and real embeddings rather
than mocks, and are skipped with a clear message if no server is reachable.

Coverage: exact match, wrong brand, wrong location, wrong category, wrong colour,
no match, time proximity, impossible timelines, neutral handling of unstated
signals, ambiguity detection, correct/incorrect/paraphrased ownership answers,
generic-answer rejection, evidence never leaking into responses, double-claim
prevention, and the full multi-turn workflow.

## Demo scenario

Click the first button in the Streamlit sidebar, or type:

> **User:** I lost my black Sony headphones near the library yesterday around 4 PM.

> **Agent:** I found a likely match: black Sony headphones (handed in at Library
> Security Desk). Confidence: 89%.
>
> Before I can release it, please describe one distinctive feature of your item —
> something specific to yours that isn't in the description above. Only the owner
> would know it.

> **User:** There is a scratch on the left earcup.

> **Agent:** Match verified. Your headphones are available at Library Security
> Desk. Pickup request LF1024 has been created.

Open **Agent Debug / Workflow** to see the trace:

```
USER INPUT
  ↓
Extract item information [rules] -> category=headphones, brand=Sony, color=black, location=library, time=yesterday 4 PM
  ↓
search_matches() -> 5 hits, top similarity = 0.76
  ↓
Confidence = 0.89 (HIGH)
  ↓
CONFIDENCE_ROUTER -> VERIFY_OWNERSHIP
  ↓
verify_ownership() [token-overlap] -> PASS (1.00)
  ↓
create_pickup_request() -> LF1024
  ↓
notify_user()
```

**Worth demonstrating alongside it** (the other sidebar buttons):

- *"I lost black **Bose** headphones near the library."* — matches the Bose item
  instead; the Sony pair drops down the ranking.
- *"I lost something."* — refuses to search, asks for more information.
- Answer the ownership question with *"it's black"* — rejected by the
  generic-answer guard, escalated to staff.

## Future improvements

Honest about what a prototype leaves out:

- **Authentication and identity.** Pickup requests are not bound to a verified
  person. In production, verification should be tied to a signed-in account and
  a staff-side ID check at collection.
- **Rate limiting on verification.** Nothing currently stops repeated guessing
  against one item. Attempts per user per item should be capped, and a burst of
  failures should escalate on its own.
- **Reverse matching.** `lost_items` are stored and embedded but not yet
  re-matched when a new item is handed in. Registering a found item should search
  open lost reports and notify likely owners.
- **Learned confidence weights.** The weights are hand-set. With enough resolved
  claims they should be fitted against ground truth.
- **Better time reasoning.** The parser handles common relative expressions; a
  proper temporal grammar (or a tool-calling LLM) would handle "last Tuesday
  after my 3pm lecture".
- **Images.** A photo of a lost item and CLIP-style multimodal retrieval would
  outperform text for visually distinctive items.
- **MCP server.** The tools are already MCP-shaped; a FastMCP wrapper would let
  other agents use this desk.
- **Staff UI.** Escalations and found-item registration are API-only today.
