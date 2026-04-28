import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Radiology Prior Relevance Classifier")

# In-memory cache keyed by (current_study_id, prior_study_id)
_cache: dict[tuple[str, str], bool] = {}

MODALITIES = {"MRI", "CT", "XR", "US", "PET", "NM", "XRAY"}
STOP_WORDS = {
    "WITH", "WITHOUT", "LIMITED", "COMPLETE", "AND", "OR",
    "LATERAL", "AP", "PA", "CONTRAST", "VIEWS", "VIEW",
    "STROKE", "SCREENING", "DIAGNOSTIC", "ROUTINE", "SERIES",
}


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


# --- Parsing helpers ---

def parse_modality_and_body(description: str) -> tuple[Optional[str], Optional[str]]:
    tokens = description.upper().split()
    modality: Optional[str] = None
    body_tokens: list[str] = []

    for i, token in enumerate(tokens):
        clean = token.rstrip(",;:")
        if clean in MODALITIES:
            modality = clean
            # Collect body part tokens after modality until a stop word
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
        for prior in case.prior_studies:
            cache_key = (case.current_study.study_id, prior.study_id)
            if cache_key in _cache:
                result = _cache[cache_key]
            else:
                result = is_relevant(case.current_study, prior)
                _cache[cache_key] = result

            predictions.append(
                Prediction(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=result,
                )
            )

    return PredictResponse(predictions=predictions)
