# Setup and Running — step by step

Written for macOS, since that is what this repo is checked out on. Windows and
Linux equivalents are at the bottom.

---

## Answers to the three questions people actually have

**How many virtual environments do I need?**
**One.** Not one per folder. `backend/requirements.txt` and
`frontend/requirements.txt` are two lists installed into the *same* venv. They
are separate files so the dashboard could later be deployed on its own against
a remote API — not because they need separate environments.

**Which folder do I run commands from?**
**Always the repository root** — `~/Documents/GitHub/Employee-Burnout-RAG`.
Never `cd backend` or `cd frontend`. Every path in `.env` is relative to the
root, and `uvicorn backend.app:app` only resolves from there.

**How many terminals?**
**Two** for a working system, **three** if you want AI-generated explanations.
They stay open. Each runs one long-lived process.

| Terminal | Runs | Stays open? |
|---|---|---|
| 1 | FastAPI backend | yes |
| 2 | Streamlit dashboard | yes |
| 3 | Ollama (optional) | yes |

---

## One-time setup

Do this once. About 5 minutes, mostly downloads.

### Terminal 1

```bash
cd ~/Documents/GitHub/Employee-Burnout-RAG
```

**Create the virtual environment** — this makes a `venv/` folder in the repo
root. It is git-ignored, so it will not be committed.

```bash
python3 -m venv venv
```

**Activate it.** Your prompt should now start with `(venv)`. If it doesn't,
activation failed and everything after this will install to the wrong place.

```bash
source venv/bin/activate
```

**Install both dependency lists into that one environment:**

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

This pulls `torch` (~200 MB) as a dependency of `sentence-transformers`, so it
is the slow step. Grab a coffee.

**Create your `.env`** — already done for you, but if it is missing:

```bash
cp .env.example .env
```

No editing needed. The default is `LLM_PROVIDER=ollama`, which needs no API key.

---

## Verify before running anything

Still in Terminal 1, still with `(venv)` active:

```bash
python backend/data_loader.py
```

**Expect:**

```
record_count            : 10
overall_burnout_percent : 10.0
risk_distribution       : {"Low": 9, "Medium": 1, "High": 0}
sentiment_distribution  : {"Negative": 6, "Positive": 4}
dominant_emotion        : joy
top high risk           : EMP_0006  Medium  score=3 (25%)
```

You will also see a warning that `sentence_count` is absent. **That is
expected** — it is the Person 2 bug documented in `docs/PERSON2_FIXES.md`, and
the backend handles it.

If this command works, your integration with Person 2's data is sound and
everything downstream will run.

---

## Running — every time

### Terminal 1 — backend

```bash
cd ~/Documents/GitHub/Employee-Burnout-RAG
source venv/bin/activate
uvicorn backend.app:app --reload --port 8000
```

**Expect:**

```
Employee Burnout RAG API v1.0.0 starting
Provider: ollama / llama3.1
Vector store ready: 10 records (rebuilt)
Uvicorn running on http://0.0.0.0:8000
```

The **first** start downloads the ~80 MB embedding model from HuggingFace and
takes 30–60 seconds. Later starts are near-instant, and say `already current`
instead of `rebuilt` because the index is reused.

Leave this running. Check it at <http://localhost:8000/docs>.

### Terminal 2 — dashboard

Open a **new** terminal window (`Cmd+T`). You must activate the venv again —
each terminal is independent.

```bash
cd ~/Documents/GitHub/Employee-Burnout-RAG
source venv/bin/activate
streamlit run frontend/app.py
```

A browser opens at <http://localhost:8501>. The sidebar should show
**Backend: 🟢 online**.

**You can stop here.** Everything works: all charts, tables, search, filters,
and the RAG panel returning rule-based answers with a red "degraded" banner.
That is a perfectly demonstrable state.

### Terminal 3 — Ollama (optional, for AI-generated explanations)

Without this, the RAG panel returns Person 2's rule-based findings, clearly
labelled as degraded. With it, you get real generated analysis.

Open a **third** terminal:

```bash
brew install ollama
ollama pull llama3.1        # ~4.7 GB, one time
ollama serve
```

Leave `ollama serve` running. Then in the dashboard, click **Refresh Database**
and ask a question. The red banner disappears and `degraded` becomes `false`.

No Homebrew? Download from <https://ollama.com/download/mac>.

---

## Stopping

`Ctrl+C` in each terminal. To leave the venv: `deactivate`.

---

## Prefer no third terminal?

Use a cloud provider instead of Ollama. Edit `.env`:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key
```

Restart Terminal 1. That is the whole change — no code edits. Gemini works the
same way with `LLM_PROVIDER=gemini` and `GOOGLE_API_KEY`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: uvicorn` | venv not active in this terminal | `source venv/bin/activate` |
| `ModuleNotFoundError: No module named 'backend'` | running from inside `backend/` | `cd` back to the repo root |
| Sidebar shows **Backend: 🔴 offline** | Terminal 1 not running | start uvicorn; charts still work meanwhile |
| Red "degraded" banner | Ollama not running | `ollama serve`, or switch provider in `.env` |
| `Person 2's burnout output was not found` | CSV missing or `BURNOUT_CSV_PATH` wrong | check `outputs/burnout_scores.csv` exists |
| First `/query` takes ~60s | model loading on first call | normal; later calls are fast |
| Port 8000 already in use | something else on that port | `uvicorn ... --port 8001` and set `API_BASE_URL=http://localhost:8001` in `.env` |
| `warning: sentence_count absent` | Person 2's bug | expected; see `docs/PERSON2_FIXES.md` |

---

## Windows (PowerShell)

Same structure, three differences: `py -3.11` instead of `python3`, backslashes,
and a different activate command.

```powershell
cd $HOME\Documents\GitHub\Employee-Burnout-RAG

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
# If blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
Copy-Item .env.example .env

# Terminal 1
uvicorn backend.app:app --reload --port 8000
# Terminal 2
streamlit run frontend\app.py
# Terminal 3 (optional) — install from https://ollama.com/download/windows
ollama serve
```

## Linux

```bash
cd ~/Employee-Burnout-RAG

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
cp .env.example .env

# Terminal 1
uvicorn backend.app:app --reload --port 8000
# Terminal 2
streamlit run frontend/app.py
# Terminal 3 (optional)
curl -fsSL https://ollama.com/install.sh | sh && ollama pull llama3.1 && ollama serve
```

---

## Hardware note

Default is `EMBEDDING_DEVICE=cpu`, which is correct everywhere and fine for a
10-row dataset. If you scale up to the full Enron corpus:

- **Apple Silicon** — set `EMBEDDING_DEVICE=mps` in `.env` (~3x faster)
- **NVIDIA GPU** — set `EMBEDDING_DEVICE=cuda`

If the hardware is not available, the backend logs a warning and falls back to
CPU rather than crashing, so committing either value is safe for the whole team.
