# Note for Person 2 — corrected after seeing the real datasets

> **CORRECTION (important).** An earlier version of this document claimed
> `df["message"]` was a straightforward bug because the column "does not
> exist". That was wrong, and it was written before `datasets/processed/` was
> available.
>
> `enron_clean.csv` **does** have a `message` column — its columns are
> `file, message, clean_text` — and because `roberta_sentiment.py` keeps every
> column it reads, `message` propagates all the way down to
> `behavior_features.py`. **On the real Enron data the pipeline does not crash.**
>
> The crash only happens when the pipeline is run against `datasets/sample.csv`,
> which has `clean_text` only. That is what produced the committed CSVs, which
> is why `sentence_count` is missing from them.
>
> There is still a real problem, but it is a **feature-consistency** issue
> rather than a crash. See "The actual issue" below. The rest of this document
> has been rewritten accordingly.

---

## The actual issue: three features are measured on cleaned text, four on raw

`models/features/behavior_features.py` draws from two different sources:

```python
df["word_count"]        = df["clean_text"]...   # cleaned
df["char_count"]        = df["clean_text"]...   # cleaned
df["avg_word_length"]   = df["clean_text"]...   # cleaned

df["sentence_count"]    = df["message"]...      # RAW, includes headers
df["exclamation_count"] = df["message"]...      # RAW
df["question_count"]    = df["message"]...      # RAW
df["uppercase_ratio"]   = df["message"]...      # RAW
```

`message` is the complete raw email, headers included:

```
Message-ID: <18782981.1075855378110.JavaMail.evans@thyme>
Date: Mon, 14 May 2001 16:39:00 -0700 (PDT)
From: phillip.allen@enron.com
X-From: Phillip K Allen
```

**Why this matters for the burnout score.** `uppercase_ratio` is computed over
text dominated by `Message-ID`, `JavaMail`, `PDT`, `X-From`, `MIME` — all
capital-heavy boilerplate that has nothing to do with how the person wrote.
That systematically inflates the ratio on every single email, and
`burnout_score.py` then applies:

```python
if row["uppercase_ratio"] > 0.25:
    score += 1
```

### Measured on the first 2,000 Enron emails

| Feature | From `message` (raw) | From `clean_text` |
|---|---|---|
| `sentence_count` mean | **27.6** | 1.0 |
| ...exceeding the `> 8` threshold | **100.0%** | 0.0% |
| `uppercase_ratio` mean | 0.152 | 0.000 |
| ...exceeding the `> 0.25` threshold | **8.2%** | 0.0% |

**Every single email crosses the `sentence_count > 8` threshold** — because
full stops in `Message-ID`, email addresses and timestamps are counted as
sentence boundaries. So `burnout_score.py` adds `+1` to *all 517,401 records*,
shifting the entire distribution up by one point.

That is not a rounding error. The bands are `≤2 Low`, `≤5 Medium`. A uniform
`+1` pushes every genuinely-Low email scoring 2 into **Medium**, and the
organisation-wide burnout rate is over-reported across the board.

### But `clean_text` is not the right answer either

Person 1's cleaning strips punctuation, so on `clean_text`:

- `sentence_count` is always ~1 (no full stops survive)
- `exclamation_count` is always 0
- `question_count` is always 0
- `uppercase_ratio` is always 0 (text is lowercased)

Four of the seven behavioural features become **dead constants**, contributing
nothing to the score.

### The genuinely correct fix

Person 1 should produce a cleaned column that **strips email headers but keeps
sentence punctuation and original casing**. Then all seven behavioural features
become meaningful, and Person 2's code can point at a single consistent source.

Until then, `clean_text` is the safer of the two: dead features are harmless,
whereas a universal `+1` actively corrupts every risk level.

**The fix is still the same one-word change** — point all seven features at
`clean_text` — but the reason is consistency and score validity, not a crash.

---

## A second observation, for Person 1

`clean_text` still contains the email headers; they have only been lowercased
and stripped of punctuation:

```
"message id date mon may pdt subject mime version content type text plain
 charset u ascii content transfer encoding bit x phillip k allen x tim belden"
```

Every one of those tokens is fed to RoBERTa and matched against the NRC
lexicon. Header boilerplate is identical across all 517,401 emails, so it adds
a constant noise floor to sentiment and emotion counts and dilutes the real
signal. Stripping everything above the first blank line would materially
improve every downstream number.

---

## Original note (still applies when running on sample.csv)

Found while building the RAG backend. **Nothing here has been applied to your
code** — this is a report and an optional patch, yours to accept or reject.

The backend works fine without it. This is about `models/` being runnable, not
about unblocking anything downstream.

---

## The problem

Two scripts in the pipeline raise `KeyError` if run today:

```
$ python models/features/behavior_features.py
KeyError: 'message'

$ python models/burnout/burnout_score.py
KeyError: 'sentence_count'
```

### Cause

`models/features/behavior_features.py` reads `df["message"]` on lines 27, 33,
37 and 56. That column does not exist. The upstream file it loads,
`outputs/emotion_features.csv`, carries only `clean_text` — `emotion_analysis.py`
never creates a `message` column, and neither does `roberta_sentiment.py`.

The second error is a **downstream symptom of the first**. Because
`behavior_features.py` dies before writing, `sentence_count` never reaches
`employee_features.csv`, so `burnout_score.py` fails when it reads
`row["sentence_count"]` on lines 38 and 93.

**`burnout_score.py` needs no edit.** Fixing the first file fixes both.

---

## Why the committed CSVs still look fine

They were generated by an earlier version of the code. Evidence:

- `outputs/behavior_features.csv` has `exclamation_count`, `question_count` and
  `uppercase_ratio` but **no `sentence_count`** — consistent with a version that
  used `clean_text` for some columns and never produced that one.
- `outputs/burnout_scores.csv` contains `"Sadness-related words found"` and
  `"Positive emotional indicators"`, but the current `explanation()` function
  emits `"Sadness detected"` and `"Positive emotion detected"`.

So the committed data is real, but it does not match the code that is committed
alongside it. Anyone who re-runs the pipeline to reproduce your results hits the
crash instead.

---

## The fix

Four occurrences of `df["message"]` become `df["clean_text"]` in
`models/features/behavior_features.py`. Nothing else changes.

```bash
git apply docs/person2_fixes.patch
```

Or by hand — lines 27, 33, 37, 56:

```python
# line 27
- df["sentence_count"] = df["message"].apply(
+ df["sentence_count"] = df["clean_text"].apply(

# line 33
- df["exclamation_count"] = df["message"].apply(
+ df["exclamation_count"] = df["clean_text"].apply(

# line 37
- df["question_count"] = df["message"].apply(
+ df["question_count"] = df["clean_text"].apply(

# line 56
- df["uppercase_ratio"] = df["message"].apply(uppercase_ratio)
+ df["uppercase_ratio"] = df["clean_text"].apply(uppercase_ratio)
```

---

## Verified before sending

Tested on a copy of `models/`, not on your files.

| Check | Result |
|---|---|
| `behavior_features.py` runs | now writes 7 columns including `sentence_count` |
| `fusion.py` runs | writes `employee_features.csv` |
| `burnout_score.py` runs | writes `burnout_scores.csv` |
| Burnout scores | **unchanged** — `0, 2, 2, 2, 0, 3, 0, 2, 0, 2` |
| Risk distribution | **unchanged** — Low 9, Medium 1, High 0 |
| Backend against the new CSV | works with **zero** changes |

The scores not moving is the important part: this is a crash fix, not a
behaviour change. Your results stand.

---

## After applying

Regenerate the outputs so code and data agree:

```bash
python models/features/behavior_features.py
python models/fusion/fusion.py
python models/burnout/burnout_score.py
```

`explanation` strings will change to match your current code
(`"Sadness detected"` rather than `"Sadness-related words found"`). Expected —
the CSVs were stale, not the code.

Then rebuild the backend index so it picks up the new file:

```bash
curl -X POST localhost:8000/refresh -d '{"force":true}' -H 'Content-Type: application/json'
```

Or just click **Refresh Database** in the dashboard sidebar.

---

## Two smaller things, no patch attached

**1. `roberta_sentiment.py` reads a file that isn't in the repo.**

```python
INPUT_FILE = "datasets/processed/enron_clean.csv"
```

`datasets/processed/` does not exist here — only `datasets/sample.csv` (the
10 rows the committed outputs derive from). Either commit the processed Enron
file, add it to the README as a manual step, or point `INPUT_FILE` at
`datasets/sample.csv` for reproducibility.

**2. `models/utils/helpers.py` is empty (0 bytes).** Harmless, but worth either
filling or deleting so the next reader doesn't go looking for something.

`models/utils/loader.py` is genuinely useful and the backend reuses it directly
rather than reimplementing `pd.read_csv` — `backend/data_loader.py` calls your
`load_data()`, so your `clean_text` validation runs on every read the API does.
