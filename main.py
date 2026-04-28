import json
import logging
import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Radiology Prior Relevance Classifier")

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
_cache: dict[tuple[str, str], bool] = {}

MODALITIES = {"MRI", "CT", "XR", "US", "PET", "NM", "XRAY"}
STOP_WORDS = {
    "WITH", "WITHOUT", "LIMITED", "COMPLETE", "AND", "OR",
    "LATERAL", "AP", "PA", "CONTRAST", "VIEWS", "VIEW",
    "STROKE", "SCREENING", "DIAGNOSTIC", "ROUTINE", "SERIES",
}

_SYSTEM = (
    "You are a radiologist assistant. Given a current examination and a list of prior "
    "examinations for the same patient, decide which priors are relevant for the radiologist "
    "to review. A prior is relevant if it shows the same or related anatomy with the same or "
    'related modality. Return ONLY valid JSON, no markdown, no explanation: '
    '{"predictions": [{"study_id": "...", "is_relevant": true}]}'
)


# --- Pydantic models ---

class StudyInfo(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class CaseInput(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: StudyInfo
    prior_studies: list[StudyInfo]


class PredictRequest(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str
    cases: list[CaseInput]


class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[Prediction]


# --- Heuristic fallback ---

def parse_modality_and_body(description: str) -> tuple[Optional[str], Optional[str]]:
    tokens = description.upper().split()
    modality: Optional[str] = None
    body_tokens: list[str] = []

    for i, token in enumerate(tokens):
        clean = token.rstrip(",;:")
        if clean in MODALITIES:
            modality = clean
            for j in range(i + 1, len(tokens)):
                word = tokens[j].rstrip(",;:")
                if word in STOP_WORDS or word in MODALITIES:
                    break
                body_tokens.append(word)
            break

    body = " ".join(body_tokens) if body_tokens else None
    return modality, body


def is_relevant(current: StudyInfo, prior: StudyInfo) -> bool:
    cur_modality, cur_body = parse_modality_and_body(current.study_description)
    pri_modality, pri_body = parse_modality_and_body(prior.study_description)

    if not cur_modality or not pri_modality:
        return False
    if not cur_body or not pri_body:
        return cur_modality == pri_modality

    return cur_modality == pri_modality and cur_body == pri_body


# --- Groq classification ---

def _groq_classify(current: StudyInfo, priors: list[StudyInfo]) -> dict[str, bool]:
    """One batched Groq call for all priors of a single current study."""
    priors_text = "\n".join(
        f"  - study_id={p.study_id}  desc={p.study_description}"
        for p in priors
    )
    user_msg = (
        f"Current examination: {current.study_description}\n\n"
        f"Prior examinations:\n{priors_text}"
    )

    completion = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = completion.choices[0].message.content or ""
    data = json.loads(raw)
    return {item["study_id"]: item["is_relevant"] for item in data["predictions"]}


def _classify_case(case: CaseInput) -> list[Prediction]:
    results: dict[str, bool] = {}
    uncached: list[StudyInfo] = []

    for prior in case.prior_studies:
        key = (case.current_study.study_id, prior.study_id)
        if key in _cache:
            results[prior.study_id] = _cache[key]
        else:
            uncached.append(prior)

    if uncached:
        try:
            llm_results = _groq_classify(case.current_study, uncached)
            for prior in uncached:
                val = llm_results.get(prior.study_id, is_relevant(case.current_study, prior))
                _cache[(case.current_study.study_id, prior.study_id)] = val
                results[prior.study_id] = val
        except Exception as exc:
            logger.warning("Groq call failed (%s), falling back to heuristics", exc)
            for prior in uncached:
                val = is_relevant(case.current_study, prior)
                _cache[(case.current_study.study_id, prior.study_id)] = val
                results[prior.study_id] = val

    return [
        Prediction(
            case_id=case.case_id,
            study_id=prior.study_id,
            predicted_is_relevant=results[prior.study_id],
        )
        for prior in case.prior_studies
    ]


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    total_priors = sum(len(c.prior_studies) for c in request.cases)
    logger.info(
        "Request received | timestamp=%s case_count=%d total_prior_count=%d",
        datetime.utcnow().isoformat(),
        len(request.cases),
        total_priors,
    )
    predictions: list[Prediction] = []
    for case in request.cases:
        predictions.extend(_classify_case(case))
    return PredictResponse(predictions=predictions)
