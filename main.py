import json
import logging
import os
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Radiology Prior Relevance Classifier")

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
_cache: dict[tuple[str, str], bool] = {}

# --- Modality and body part maps for heuristic fallback ---

MODALITY_ALIASES: dict[str, str] = {
    "MRI": "MRI", "MR": "MRI",
    "CT": "CT", "CAT": "CT",
    "XR": "XR", "XRAY": "XR", "X-RAY": "XR", "CXR": "XR", "KUB": "XR",
    "US": "US", "ULTRASOUND": "US", "SONO": "US", "SONOGRAM": "US",
    "PET": "PET",
    "NM": "NM", "NMED": "NM", "NUCLEAR": "NM", "SPECT": "NM",
    "MAM": "MAM", "MAMMO": "MAM", "MAMMOGRAPHY": "MAM", "MAMMOGRAM": "MAM",
    "ECHO": "ECHO", "ECHOCARDIOGRAM": "ECHO", "ECHOCARDIOGRAPHY": "ECHO",
}

# Two-token patterns that must be detected before single-token scan
_MULTI_PATTERNS: list[tuple[str, str]] = [
    ("BONE SCAN", "NM"),
    ("BONE SPECT", "NM"),
    ("PET/CT", "PET"),
    ("PET CT", "PET"),
    ("X-RAY", "XR"),
]

# Cross-modality pairs considered relevant (same body region required)
CROSS_MODALITY: set[frozenset] = {
    frozenset({"CT", "XR"}),
    frozenset({"MRI", "CT"}),
    frozenset({"US", "CT"}),
    frozenset({"MRI", "US"}),
    frozenset({"NM", "CT"}),
    frozenset({"PET", "CT"}),
    frozenset({"PET", "NM"}),
    frozenset({"ECHO", "NM"}),  # cardiac echo ↔ cardiac NM (MUGA/stress)
    frozenset({"ECHO", "MRI"}),
}

STOP_WORDS: set[str] = {
    "WITH", "WITHOUT", "LIMITED", "COMPLETE", "AND", "OR",
    "LATERAL", "AP", "PA", "CONTRAST", "VIEWS", "VIEW",
    "STROKE", "SCREENING", "DIAGNOSTIC", "ROUTINE", "SERIES",
    "W", "W/O", "WO", "BILATERAL", "UNILATERAL",
    "RIGHT", "LEFT", "UPPER", "LOWER", "ANTERIOR", "POSTERIOR",
}

BODY_SYNONYMS: dict[str, str] = {
    "ABD": "ABDOMEN", "ABDO": "ABDOMEN", "ABDOMINAL": "ABDOMEN",
    "PELV": "PELVIS", "PELVIC": "PELVIS",
    "HEAD": "BRAIN", "NEURO": "BRAIN", "CRANIAL": "BRAIN", "INTRACRANIAL": "BRAIN",
    "THORAX": "CHEST", "THORACIC": "CHEST", "PULMONARY": "CHEST", "LUNGS": "CHEST",
    "NECK": "NECK", "CERVICAL": "NECK",
    "LUMBAR": "SPINE", "LUMBOSACRAL": "SPINE", "LS": "SPINE", "TL": "SPINE",
    "MYO": "HEART", "MYOCARDIAL": "HEART", "MYOCARDIUM": "HEART",
    "CARDIAC": "HEART", "CARDIO": "HEART",
    "HEPATIC": "LIVER", "HEP": "LIVER",
    "RENAL": "KIDNEY", "KIDNEYS": "KIDNEY",
}

_SYSTEM = (
    "You are an expert radiologist assistant. Given a current examination and a list of prior "
    "examinations for the same patient, determine which priors are RELEVANT for the radiologist to review.\n\n"
    "A prior is RELEVANT if it shows the same or related anatomy with the same or related imaging modality.\n\n"
    "MODALITY RELEVANCE RULES:\n"
    "- Same modality = relevant (MRI↔MRI, CT↔CT, etc.)\n"
    "- CT ↔ X-ray (XR/CXR/KUB/CHEST): relevant when same body area\n"
    "- MRI ↔ CT: relevant when same body area\n"
    "- Ultrasound (US/Sono) ↔ CT: relevant when same body area\n"
    "- MRI ↔ Ultrasound: relevant when same body area\n"
    "- Nuclear Medicine (NM/SPECT/bone scan) ↔ CT: relevant when same body area\n"
    "- PET ↔ CT or NM: relevant when same body area\n"
    "- Mammography (MAM/MAMMO/Mammogram): ONLY relevant to other breast/mammography studies\n"
    "- Echo (ECHO/Echocardiogram) ↔ cardiac CT, cardiac MRI, cardiac NM (MUGA/stress): relevant\n"
    "- Different body areas = NOT relevant regardless of modality\n\n"
    "BODY REGION EQUIVALENCES:\n"
    "- Head = Brain = Cranial = Neuro = Intracranial\n"
    "- Chest = Thorax = Thoracic = Pulmonary = Lungs = CXR\n"
    "- Abdomen = Abdominal = Abd\n"
    "- Abdomen and Abdomen/Pelvis = overlapping (relevant to each other)\n"
    "- Pelvis = Pelvic\n"
    "- Spine subtypes overlap: cervical / thoracic / lumbar / lumbosacral / LS spine\n"
    "- Heart = Cardiac = Cardio = Myocardium = Myocardial = MUGA\n"
    "- Kidney = Renal\n\n"
    "MODALITY RECOGNITION:\n"
    "- 'CHEST' alone (without MRI/CT/US qualifier) = Chest X-ray\n"
    "- 'BONE SCAN' or 'BONE SPECT' = Nuclear Medicine (NM)\n"
    "- 'MAM', 'MAMMO', 'MAMMOGRAPHY', 'MAMMOGRAM' = Mammography\n"
    "- 'ECHO', 'ECHOCARDIOGRAM' = Cardiac Ultrasound\n"
    "- 'PET/CT' = PET scan\n\n"
    "Return ONLY valid JSON, no markdown, no explanation:\n"
    '{"predictions": [{"study_id": "...", "is_relevant": true}]}'
)

_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "study_id": {"type": "string"},
                    "is_relevant": {"type": "boolean"},
                },
                "required": ["study_id", "is_relevant"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["predictions"],
    "additionalProperties": False,
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


# --- Heuristic fallback ---

def _parse_modality_and_body(description: str) -> tuple[Optional[str], Optional[str]]:
    text = description.upper()

    modality: Optional[str] = None
    mod_token_end = 0

    # Check multi-word patterns first
    for pattern, mod in _MULTI_PATTERNS:
        if pattern in text:
            modality = mod
            idx = text.find(pattern)
            mod_token_end = idx + len(pattern)
            break

    # Single-token modality scan
    if modality is None:
        tokens = text.split()
        for i, token in enumerate(tokens):
            clean = token.rstrip(",;:./")
            if clean in MODALITY_ALIASES:
                modality = MODALITY_ALIASES[clean]
                mod_token_end = sum(len(t) + 1 for t in tokens[: i + 1])
                break

    if modality is None:
        # "CHEST" alone as first meaningful word = XR
        tokens = text.split()
        if tokens and tokens[0].rstrip(",;:.") == "CHEST":
            modality = "XR"
            mod_token_end = len(tokens[0]) + 1

    if modality is None:
        return None, None

    # Extract body part tokens from the remainder
    remainder = text[mod_token_end:].split()
    body_tokens: list[str] = []
    for word in remainder:
        clean = word.rstrip(",;:./")
        if not clean or clean in STOP_WORDS or clean in MODALITY_ALIASES:
            continue
        normalized = BODY_SYNONYMS.get(clean, clean)
        body_tokens.append(normalized)
        if len(body_tokens) >= 2:
            break

    body = " ".join(body_tokens) if body_tokens else None
    return modality, body


def _heuristic_relevant(current: StudyInfo, prior: StudyInfo) -> bool:
    cur_mod, cur_body = _parse_modality_and_body(current.study_description)
    pri_mod, pri_body = _parse_modality_and_body(prior.study_description)

    if not cur_mod or not pri_mod:
        return False

    mod_pair = frozenset({cur_mod, pri_mod})
    same_mod = cur_mod == pri_mod
    cross_ok = mod_pair in CROSS_MODALITY

    if not same_mod and not cross_ok:
        return False

    # Mammography: only relevant to itself
    if "MAM" in {cur_mod, pri_mod} and cur_mod != pri_mod:
        return False

    if not cur_body or not pri_body:
        return same_mod  # no body info; be conservative on cross-modality

    return cur_body == pri_body


# --- Anthropic classification ---

def _anthropic_classify(current: StudyInfo, priors: list[StudyInfo]) -> dict[str, bool]:
    priors_text = "\n".join(
        f"  - study_id={p.study_id}  desc={p.study_description}  date={p.study_date}"
        for p in priors
    )
    user_msg = (
        f"Current examination: {current.study_description} (date: {current.study_date})\n\n"
        f"Prior examinations:\n{priors_text}"
    )

    response = _client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        messages=[{"role": "user", "content": user_msg}],
    )

    # Extract text block (thinking blocks may precede it)
    raw = next((b.text for b in response.content if b.type == "text"), "")
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
            llm_results = _anthropic_classify(case.current_study, uncached)
            for prior in uncached:
                val = llm_results.get(prior.study_id, _heuristic_relevant(case.current_study, prior))
                _cache[(case.current_study.study_id, prior.study_id)] = val
                results[prior.study_id] = val
        except Exception as exc:
            logger.warning("Anthropic call failed (%s), falling back to heuristics", exc)
            for prior in uncached:
                val = _heuristic_relevant(case.current_study, prior)
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
