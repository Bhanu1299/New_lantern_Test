# Radiology Prior Relevance Classifier

A FastAPI service that classifies which prior radiology studies are relevant for comparison when reading a new exam. Achieves **97% accuracy** on the competition evaluation dataset.

## How It Works

The service uses a two-layer approach:

1. **Anthropic LLM (primary)** — `claude-opus-4-7` with a domain-expert system prompt classifies all prior studies for each case. Uses prompt caching for efficiency.
2. **Heuristic parser (fallback)** — A rule-based modality + body region parser handles cases when the API call fails. Achieves 93.9% accuracy standalone.

### Relevance Rules

A prior exam is **relevant** when it images the same or overlapping body region with a compatible modality:

| Modality Pairs | Example |
|---|---|
| CT ↔ X-ray | Chest CT vs chest XR |
| MRI ↔ CT | Brain MRI vs head CT |
| US ↔ CT or MRI | Renal US vs CT abdomen |
| NM ↔ CT or PET | Bone scan vs chest CT |
| PET ↔ CT or NM | PET/CT wholebody vs any CT |
| Echo ↔ cardiac MRI / cardiac NM | TTE vs cardiac NM |
| Mammography ↔ breast US / breast MRI | MAM screen vs US breast |

Laterality is respected: right-sided mammograms are **not** relevant to left-sided mammograms.

---

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /predict`

**Request:**
```json
{
  "challenge_id": "string",
  "schema_version": 1,
  "generated_at": "2026-01-01T00:00:00Z",
  "cases": [
    {
      "case_id": "case_001",
      "patient_id": "pt_001",
      "patient_name": "John Doe",
      "current_study": {
        "study_id": "s1",
        "study_description": "CT CHEST WITH CONTRAST",
        "study_date": "2026-01-01"
      },
      "prior_studies": [
        {
          "study_id": "s2",
          "study_description": "XR chest 2V PA/lat",
          "study_date": "2025-06-01"
        }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "case_id": "case_001",
      "study_id": "s2",
      "predicted_is_relevant": true
    }
  ]
}
```

---

## Setup

### Requirements
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
anthropic>=0.49.0
```

### Environment Variables
| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `PORT` | Railway auto-sets | Server port |

### Run Locally
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --port 8000
```

### Deploy (Railway)
Push to `main` branch — Railway auto-deploys via the `Procfile`:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Local Scoring

### Heuristic only (instant, no API calls)
```bash
python3 score_heuristic.py
```

### Full API scoring (requires local server running)
```bash
uvicorn main:app --port 8000 &
python3 score_local.py
```

---

## Architecture

```
POST /predict
    │
    ├─ For each case: check in-memory cache
    │
    ├─ Uncached priors → _anthropic_classify()
    │       claude-opus-4-7
    │       effort=high
    │       System prompt cached (ephemeral)
    │
    └─ On API failure → _heuristic_relevant()
            Parse modality + body region
            Match via CROSS_MODALITY table
            Apply body overlap rules
```

### Heuristic Parser Features
- Multi-word pattern detection (`BONE SCAN`, `PET/CT`, `PET-CT`)
- 90+ modality aliases (`TTE`, `TEE`, `MRA`, `DXA`, `NMmyo*`, etc.)
- 60+ body synonyms (`CERV`→NECK, `PLV`→PELVIS, `UROGRAM`→KIDNEY, etc.)
- 50+ stop words filtered during body extraction
- MAM laterality: `BREAST_RT` ≠ `BREAST_LT`
- PET wholebody detection → always relevant to any compatible CT/NM
- Body overlaps: KIDNEY↔ABDOMEN, LIVER↔ABDOMEN, PELVIS↔ABDOMEN
- Standalone body names detected as plain XR (`Abdomen` → XR+ABDOMEN)

---

## Results

| Metric | Score |
|---|---|
| Competition accuracy | **97%** |
| Heuristic-only accuracy | 93.9% |
| Heuristic precision | 91.9% |
| Heuristic recall | 81.4% |
| Dataset size | 27,614 pairs across 996 cases |
