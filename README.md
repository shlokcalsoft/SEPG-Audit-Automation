# SEPG Audit Automation

Automatically fills the SEPG Master Audit Template (`.xlsx`) with compliance verdicts and timestamped evidence observations derived from a meeting transcript (`.docx`) using Groq's free-tier LLMs.

---

## What it does

1. **Reads** the 248-checkpoint audit template (`Q2 2026` sheet, column L for verdicts, column Q for observations).
2. **Parses** the meeting transcript into speaker-tagged, timestamped segments.
3. **Retrieves** the best-matching transcript excerpts for each checkpoint using hybrid TF-IDF + keyword overlap scoring.
4. **Assesses** compliance via Groq LLM — one API call per Activity group (43 batches total).
5. **Writes** verdicts to column L and evidence observations to column Q in the format:
   ```
   [HH:MM:SS | Speaker] concise evidence-based sentence.
   ```
6. **Generates** a review log flagging low-confidence verdicts, evidence mismatches, and unresolved checkpoints.

---

## Prerequisites

- Python 3.10+
- A [Groq free-tier account](https://console.groq.com) with an API key

---

## Setup

### 1. Clone / open the project
```powershell
cd D:\SEPG_Better_Version\SEPG_Audit_Automation
```

### 2. Create & activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Set your Groq API key
Edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Place input files in `data/`
| File | Description |
|---|---|
| `data/template.xlsx` | Copy of the SEPG Master Audit Template |
| `data/sample_transcript.docx` | Sanitized meeting transcript |

```powershell
Copy-Item "path\to\SEPG_Master_Audit_Template.xlsx" "data\template.xlsx"
Copy-Item "path\to\transcript.docx" "data\sample_transcript.docx"
```

---

## Running the pipeline

```powershell
# From project root with venv active:
python src/pipeline.py
```

The pipeline will:
- Print progress for each of the 43 activity batches
- Rate-limit automatically (sleeps between Groq calls as needed)
- Take approximately 15–45 minutes depending on transcript length and rate limits

---

## Output files

Both files appear in the `output/` directory, named with the run timestamp:

| File | Description |
|---|---|
| `output/YYYYMMDD_HHMMSS_scorecard.xlsx` | Filled copy of the template — verdicts in col L, observations in col Q |
| `output/YYYYMMDD_HHMMSS_review_log.json` | Checkpoints flagged for human review |

**The original `data/template.xlsx` is never modified.**

---

## Valid verdict values

These must match the template's N-column scoring formula exactly (case-sensitive):

| Verdict | Score | Meaning |
|---|---|---|
| `Yes` | 5 | Fully compliant |
| `Partial` | 4 | Partially compliant |
| `In Progress` | 3 | Actively being worked on |
| `Not clarified` | 2 | Insufficient evidence (**lowercase 'c'**) |
| `No` | 0 | Not compliant |
| `NA` | — | Not applicable — **cell left completely blank** |

---

## Observation column format

Every non-NA verdict gets an observation in column Q:
```
[HH:MM:SS | Speaker] Concise evidence-based sentence.
```
Example:
```
[00:14:32 | Project Manager] SOW was reviewed and approved by the client.
```
The timestamp lets a human reviewer jump directly to the relevant part of the transcript.

---

## Configuration (`config.yaml`)

All tunable values live in `config.yaml` — nothing is hardcoded in Python:

| Key | Default | Description |
|---|---|---|
| `template.sheet_name` | `Q2 2026` | Checklist sheet name |
| `template.expected_checkpoint_count` | `248` | Assertion guard |
| `retrieval.tfidf_weight` | `0.70` | TF-IDF cosine similarity weight |
| `retrieval.keyword_weight` | `0.30` | Keyword overlap weight |
| `retrieval.top_k` | `5` | Top-K segments per checkpoint |
| `retrieval.window_radius` | `2` | ±segment context window |
| `groq.escalation_confidence_threshold` | `0.35` | Below this → use 70B model |
| `groq.rate_limits` | See config | Groq free-tier limits (update from console) |
| `review.flag_confidence_threshold` | `0.4` | Below this → flag for review |

---

## Running tests

```powershell
python -m pytest tests/ -v
```

Unit tests use synthetic data and the real template. The 2 real-transcript tests and the integration test are auto-skipped until `data/sample_transcript.docx` is present.

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `AssertionError: Expected 248 checkpoints` | Wrong column mapping | Check `config.yaml` column letters against the real template |
| `GROQ_API_KEY not set` | Missing `.env` | Add your key to `.env` |
| Pipeline hangs/slow | Rate limiting | Normal — the limiter sleeps to stay under Groq free-tier caps |
| `batch_failed_twice` entries in review log | Groq API error or malformed response | Check internet connection; may re-run — failed batches leave cells blank |
| Verdict casing wrong | LLM returned wrong case | The system prompt enforces exact casing; report if persistent |
| NA cell has content | Critical bug | Report immediately — should never happen due to two enforcement points |
