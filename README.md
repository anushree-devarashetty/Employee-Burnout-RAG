# Explainable Employee Burnout Detection using ML, Sentiment Analysis and RAG

Detects burnout signals in workplace communication and explains them in plain
language, using machine learning for the scoring and Retrieval-Augmented
Generation for the explanation.

Existing methods rely on surveys or black-box models. This system analyses
communication continuously and — critically — shows its working: every number
traces back to a transparent rule, and every generated recommendation is
grounded in specific retrieved records that the user can inspect.

---

## Architecture

```
datasets/                 Person 1  — collection and preprocessing
   |
   v
models/                   Person 2  — RoBERTa sentiment, NRC emotions,
   |                                   behaviour features, burnout scoring
   v
outputs/burnout_scores.csv           <-- THE INTEGRATION CONTRACT
   |
   +----------------------------+
   |                            |
   v                            v
backend/     Person 3           frontend/   Person 4
  FastAPI + LangChain             Streamlit + Plotly + ReportLab
  Sentence-Transformers
  ChromaDB
  OpenAI | Gemini | Ollama
```

**Person 3 and Person 4 consume `outputs/burnout_scores.csv` read-only.** The
ML pipeline is never re-run by the backend or the dashboard. Row reading is
delegated to Person 2's own `models/utils/loader.py` rather than reimplemented.

`models/` and `datasets/` contain **zero modifications** — no files added, none
changed. Verified with `git diff --stat -- models/ datasets/ outputs/`.

### RAG pipeline

```
User Question -> Embedding -> Retriever -> Prompt -> LLM -> Structured JSON
                 MiniLM       ChromaDB    grounded  swappable  8 fixed keys
                 384-dim      cosine      + fenced  via .env   validated
```

---

## Quick start

Three terminals, or run the first two steps once and leave them.

```bash
# 1. Install
python -m venv venv && source venv/bin/activate      # see per-OS commands below
pip install -r backend/requirements.txt -r frontend/requirements.txt

# 2. Configure
cp .env.example .env

# 3. Local model (default provider — free, offline, no API key)
ollama pull llama3.1 && ollama serve

# 4. Backend                    -> http://localhost:8000/docs
uvicorn backend.app:app --reload --port 8000

# 5. Dashboard                  -> http://localhost:8501
streamlit run frontend/app.py
```

Run every command **from the repository root**, not from inside `backend/` or
`frontend/`.

---

## Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/anushree-devarashetty/Employee-Burnout-RAG.git
cd Employee-Burnout-RAG

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
# If activation is blocked:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt

Copy-Item .env.example .env

# Optional: local LLM. Download from https://ollama.com/download/windows
ollama pull llama3.1

# Terminal 1
uvicorn backend.app:app --reload --port 8000
# Terminal 2
streamlit run frontend\app.py
```

### macOS

```bash
git clone https://github.com/anushree-devarashetty/Employee-Burnout-RAG.git
cd Employee-Burnout-RAG

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

cp .env.example .env

# Optional: local LLM
brew install ollama && ollama pull llama3.1 && ollama serve

# Apple Silicon: GPU embeddings are ~3x faster. In .env set:
#   EMBEDDING_DEVICE=mps

uvicorn backend.app:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                    # terminal 2
```

### Linux

```bash
git clone https://github.com/anushree-devarashetty/Employee-Burnout-RAG.git
cd Employee-Burnout-RAG

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

cp .env.example .env

curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1 && ollama serve &

# NVIDIA GPU: in .env set EMBEDDING_DEVICE=cuda

uvicorn backend.app:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                    # terminal 2
```

### Why not the root `requirements.txt`

The root file is a full `pip freeze` of one developer's machine: 400+ lines
including TensorFlow, OpenCV, MediaPipe, Flask and ~200 macOS-only `pyobjc`
packages. **It cannot be installed on Windows or Linux.** `backend/` and
`frontend/` each list only what they import, with lower bounds rather than
exact pins so they coexist with Person 2's `torch`/`transformers` stack.

---

## Switching LLM provider

Edit `.env`. Change nothing else — no Python file needs touching.

| Provider | `.env` |
|---|---|
| **Ollama** (default, free, offline) | `LLM_PROVIDER=ollama`<br>`OLLAMA_MODEL=llama3.1` |
| **OpenAI** | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=sk-...`<br>`OPENAI_MODEL=gpt-4o-mini` |
| **Gemini** | `LLM_PROVIDER=gemini`<br>`GOOGLE_API_KEY=...`<br>`GEMINI_MODEL=gemini-2.0-flash` |
| **Groq / Together / vLLM** | `LLM_PROVIDER=openai`<br>`OPENAI_BASE_URL=https://api.groq.com/openai/v1` |

`backend/config.py` validates credentials for the **selected** provider only,
at startup, with an error naming the exact variable to fix. Someone running
local Ollama is never asked for an OpenAI key.

---

## API

Interactive docs at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service banner |
| GET | `/health` | Health, config, data provenance. `?probe_llm=true` also pings the provider |
| GET | `/statistics` | Aggregate metrics for the dashboard |
| POST | `/query` | Ask a question — full RAG pipeline |
| POST | `/report` | Structured report |
| POST | `/refresh` | Rebuild the vector index from the CSV |

`/statistics` and `/refresh` are additions beyond the four specified endpoints:
`/statistics` stops each chart re-reading and re-aggregating the CSV, and
`/refresh` is the server-side action behind the specified "Refresh Database"
button.

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which employees show signs of burnout, and why?","top_k":5}'
```

Always returns these eight keys under `answer`:

```json
{
  "answer": {
    "burnout_summary": "...",
    "emotional_analysis": "...",
    "possible_causes": ["..."],
    "risk_level": "Low | Medium | High",
    "suggested_interventions": ["..."],
    "hr_recommendations": ["..."],
    "mental_wellness_suggestions": ["..."],
    "confidence_note": "..."
  },
  "sources": [{ "employee_id": "EMP_0006", "similarity": 0.87, "...": "..." }],
  "degraded": false
}
```

**`degraded: true`** means the LLM was unreachable and the answer was assembled
from Person 2's rule-based `explanation` column instead. The request still
returns **HTTP 200** with a usable answer, and both the dashboard and the PDF
display a banner saying so. An HR user is never shown a stack trace, and never
left unable to tell whether a language model was involved.

---

## Testing

Run each step from the repository root. Expected output is given so you can
tell pass from fail without guessing.

### 1. Configuration

```bash
python backend/config.py
```

Expect a JSON block ending with `OK Person 2's data found: .../burnout_scores.csv`.
If a provider is selected without credentials, expect a `ValidationError`
naming the variable — that is the intended behaviour, not a bug.

### 2. Data layer

```bash
python backend/data_loader.py
```

Expect, for the committed 10-row CSV:

```
record_count            : 10
overall_burnout_percent : 10.0
risk_distribution       : {"Low": 9, "Medium": 1, "High": 0}
sentiment_distribution  : {"Negative": 6, "Positive": 4}
dominant_emotion        : joy
top high risk           : EMP_0006  Medium  score=3 (25%)
```

Plus a warning that `sentence_count` is absent — expected, see *Known issues*.

### 3. Embeddings

```bash
python backend/rag/embeddings.py
```

First run downloads ~80 MB from HuggingFace. Expect `Dimension: 384`,
`L2 norm of first vector: 1.000000`, and *"I am completely exhausted after
working late."* ranking top for *"Who is burned out and overworked?"*

### 4. Vector store

```bash
python backend/rag/vectorstore.py
```

Expect `"records_indexed": 10`, `"rebuilt": true`. Run it again with
`force=False` and expect `"rebuilt": false` — the fingerprint check skipping a
needless re-embed.

### 5. Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

Expect on startup:

```
Employee Burnout RAG API v1.0.0 starting
Provider: ollama / llama3.1
Vector store ready: 10 records (rebuilt)
```

Then:

```bash
curl localhost:8000/health | python -m json.tool
curl localhost:8000/statistics | python -m json.tool
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
     -d '{"question":"Who is most at risk?","top_k":3}' | python -m json.tool
```

`/health` should report `"status": "ok"` and `"record_count": 10`.

### 6. Degraded path — test this deliberately

Stop Ollama, then re-run the `/query` call. Expect **HTTP 200**, not 500, with
`"degraded": true` and a `confidence_note` beginning `DEGRADED RESPONSE`. This
is the single most valuable test in the list: it is the failure your demo is
most likely to hit.

### 7. Dashboard

```bash
streamlit run frontend/app.py
```

Screenshots worth capturing for a report:

| # | View | Shows |
|---|---|---|
| 1 | Top of dashboard | Four metric cards + Overall Burnout gauge at 10% "Healthy" |
| 2 | Chart grid | Sentiment pie (6 Neg / 4 Pos), emotion bar (joy 3, trust 2), risk bars (Low 9, Medium 1, High 0) |
| 3 | Score histogram + scatter | Threshold lines at Medium ≥3 and High ≥6 |
| 4 | Top High-Risk table | EMP_0006 first |
| 5 | Search box | Type `EMP_0006` → exactly one row |
| 6 | RAG panel | Eight sections, amber confidence callout, evidence table with relevance scores |
| 7 | Degraded banner | Red banner with Ollama stopped |
| 8 | PDF | Generated report opened from the download |

### 8. Backend down, dashboard up

Stop uvicorn, leave Streamlit running, reload the page. **Every chart must still
render** — they read the CSV directly. Only the RAG and report panels are
unavailable, and the sidebar shows `Backend: 🔴 offline` with the start command.

---

## Verification performed

Every module was tested during development, including against a real ChromaDB
index and a live FastAPI server. Bugs found and fixed this way:

| Bug | Impact if shipped |
|---|---|
| `@lru_cache` on functions taking Pydantic `Settings` | `TypeError: unhashable type` — backend would not start |
| `collection.modify()` rejects `hnsw:space` even when unchanged | Fingerprint never persisted; full re-embed on every startup |
| `httpx` proxies `localhost` when `HTTP_PROXY` is set | Local Ollama unreachable behind a corporate proxy |
| Missing columns not materialised | `AttributeError` on `sentence_count` |
| `idxmax` on all-zero emotion rows | 6 of 10 records mislabelled **"anger"** instead of "neutral" |
| Unescaped `<`, `>`, `&` in ReportLab | A message containing `&` would fail the whole PDF |

---

## Known issues in the ML pipeline (Person 2's scope)

Found while integrating. **Not fixed here** — they belong to Person 2, and the
backend is written to tolerate them.

1. **`models/features/behavior_features.py` reads `df["message"]`** (lines 27,
   33, 37, 56), a column that does not exist in the upstream
   `emotion_features.csv`. The script raises `KeyError: 'message'` if run today.
   Fix: change `df["message"]` to `df["clean_text"]`.

2. **`models/burnout/burnout_score.py` reads `row["sentence_count"]`** (lines
   38, 93), which was never written because of issue 1. Also raises `KeyError`.

3. **The committed CSVs are stale.** `burnout_scores.csv` contains explanations
   like `"Sadness-related words found"`, but the current `explanation()`
   function emits `"Sadness detected"`. The CSVs predate the current code.

4. **`roberta_sentiment.py` reads `datasets/processed/enron_clean.csv`**, which
   is not in the repository. The committed outputs derive from
   `datasets/sample.csv` (10 rows).

The backend treats behaviour columns as optional and **does not zero-fill
them**: `sentence_count` is missing because a script crashed, not because these
messages contain zero sentences. Writing `0` would put a fabricated measurement
into the vector store and show it to the LLM as evidence.

---

## Git workflow

Four collaborators. `ml-sentiment` is currently the default branch.

### Branches

| Person | Branch | Scope |
|---|---|---|
| 1 | `data-preprocessing` | `datasets/` |
| 2 | `ml-sentiment` | `models/`, `outputs/` |
| 3 | `rag-backend` | `backend/` |
| 4 | `frontend-dashboard` | `frontend/` |

### Commits

Person 3 (`rag-backend`):

```
chore: add .env.example and populate the empty .gitignore
feat(backend): add config with per-provider credential validation
feat(backend): add data loader reusing models/utils/loader.py
feat(backend): add API schemas with the 8-key analysis contract
feat(rag): add all-MiniLM-L6-v2 embeddings with lazy thread-safe load
feat(rag): add persistent ChromaDB store with fingerprint-based rebuild
feat(rag): add top-k retriever with filter relaxation and ID detection
feat(rag): add provider-agnostic LLM client (OpenAI, Gemini, Ollama)
feat(prompts): add grounded burnout prompts and rule-based fallback
feat(rag): add pipeline with resilient JSON parsing and degradation
feat(backend): add FastAPI app with six endpoints
fix(rag): replace lru_cache singletons — Pydantic Settings is unhashable
fix(rag): persist fingerprint without hnsw:space, which modify() rejects
fix(rag): bypass HTTP proxy for loopback LLM endpoints
docs: add README with install, API, testing and workflow
```

Person 4 (`frontend-dashboard`):

```
feat(frontend): add API client that never raises into the UI
feat(frontend): add data utilities delegating to backend.data_loader
feat(frontend): add Plotly charts with empty-state handling
feat(frontend): add ReportLab PDF with confidence note above recommendations
feat(frontend): add Streamlit dashboard with sidebar controls
```

### Merge and PR order

Order matters: the dashboard imports `backend.data_loader`, so merging
`frontend-dashboard` first leaves `main` in a state where the dashboard cannot
start.

```
1. PR #1   data-preprocessing  -> main     Person 1, reviewed by Person 2
2. PR #2   ml-sentiment        -> main     Person 2, reviewed by Person 1
3. PR #3   rag-backend         -> main     Person 3, reviewed by Person 2   <- depends on #2
4. PR #4   frontend-dashboard  -> main     Person 4, reviewed by Person 3   <- depends on #3
```

Rebase before opening each PR:

```bash
git checkout rag-backend
git fetch origin
git rebase origin/main
git push --force-with-lease origin rag-backend
```

### One-time cleanup

`.DS_Store` is committed and shows as modified on every `git status`. A
`.gitignore` rule cannot untrack it:

```bash
git rm --cached .DS_Store
git commit -m "chore: untrack .DS_Store now that .gitignore covers it"
```

---

## Configuration reference

See `.env.example` for all settings with inline commentary. The ones most
worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | The only switch needed to change provider |
| `BURNOUT_CSV_PATH` | `outputs/burnout_scores.csv` | Person 2's output, read-only |
| `BURNOUT_SCORE_MAX` | `12` | Display normalisation only — never used to classify risk |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384-dim; changing this forces an index rebuild |
| `RETRIEVER_TOP_K` | `5` | Records fed to the LLM |
| `RETRIEVER_MAX_K` | `25` | Ceiling — stops a caller overflowing the context window |
| `API_TIMEOUT` | `180` | Must exceed `LLM_TIMEOUT`, or the dashboard aborts mid-generation |

---

## Responsible use

Burnout risk inferred from workplace text is an **indicative signal, not a
clinical or diagnostic finding**. The system is built so this cannot be
forgotten:

- Every analysis carries a `confidence_note`. If the model omits it, the
  backend substitutes one rather than serving the analysis without it — the
  only field treated this way.
- The confidence note appears **before** the recommendations in both the
  dashboard and the PDF, because a reader deciding what to do about a named
  person must see the caveat first.
- Prompts forbid clinical language and diagnosis, and require the model to say
  when the evidence does not support an answer.
- Retrieved employee text is fenced and labelled as data, so message content
  cannot be read as an instruction to the model.
- Every generated answer ships with the records it was based on, so a user can
  check the reasoning rather than trusting it.

Use this to start supportive conversations. Never as the sole basis for a
decision affecting an individual.
