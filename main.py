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
    "MRI": "MRI", "MR": "MRI", "FMRI": "MRI", "MRA": "MRI",
    "CT": "CT", "CAT": "CT",
    "XR": "XR", "XRAY": "XR", "X-RAY": "XR", "CXR": "XR", "KUB": "XR",
    "US": "US", "ULTRASOUND": "US", "SONO": "US", "SONOGRAM": "US",
    "PET": "PET",
    "NM": "NM", "NMED": "NM", "NUCLEAR": "NM", "SPECT": "NM",
    "DXA": "NM", "DEXA": "NM",  # bone density scan
    "MAM": "MAM", "MAMMO": "MAM", "MAMMOGRAPHY": "MAM", "MAMMOGRAM": "MAM",
    "ECHO": "ECHO", "ECHOCARDIOGRAM": "ECHO", "ECHOCARDIOGRAPHY": "ECHO",
    "TTE": "ECHO", "TEE": "ECHO",  # transthoracic/transesophageal echo
}

# Two-token patterns that must be detected before single-token scan
_MULTI_PATTERNS: list[tuple[str, str]] = [
    ("BONE SCAN", "NM"),
    ("BONE SPECT", "NM"),
    ("PET/CT", "PET"),
    ("PET CT", "PET"),
    ("X-RAY", "XR"),
]

# PET whole-body description patterns → body=None
_PET_WHOLEBODY_PATTERNS = (
    "SKULL TO THIGH", "SKULL TO MID THIGH", "SKULL-THIGH",
    "SKULL TO THIGHS", "WHOLEBODY", "WHOLE BODY", "SKULLTHIGH",
)

# Cross-modality pairs considered relevant (same body region required)
CROSS_MODALITY: set[frozenset] = {
    frozenset({"CT", "XR"}),
    frozenset({"MRI", "CT"}),
    frozenset({"US", "CT"}),
    frozenset({"MRI", "US"}),
    frozenset({"NM", "CT"}),
    frozenset({"PET", "CT"}),
    frozenset({"PET", "NM"}),
    frozenset({"ECHO", "NM"}),   # cardiac echo ↔ cardiac NM (MUGA/stress)
    frozenset({"ECHO", "MRI"}),
    frozenset({"MRI", "XR"}),    # MRI ↔ X-ray of same region (e.g. spine)
    frozenset({"MAM", "US"}),   # breast US ↔ mammography
    frozenset({"MAM", "MRI"}),  # breast MRI ↔ mammography
    frozenset({"US", "XR"}),
    frozenset({"NM", "XR"}),
}

STOP_WORDS: set[str] = {
    "WITH", "WITHOUT", "LIMITED", "COMPLETE", "AND", "OR",
    "LATERAL", "AP", "PA", "CONTRAST", "VIEWS", "VIEW",
    "STROKE", "SCREENING", "DIAGNOSTIC", "ROUTINE", "SERIES",
    "W", "W/O", "WO", "BILATERAL", "UNILATERAL",
    "RIGHT", "LEFT", "RT", "LT", "BI", "BILAT",
    "UPPER", "LOWER", "ANTERIOR", "POSTERIOR",
    "CNTRST", "CAD", "TOMO", "PERF", "REST", "STR", "STRESS", "ANGIO",
    "HEMATURIA", "COLIC", "FOR", "DOPPLER", "DOP",
    "LIMITED", "COMPLETE", "FULL", "COMPREHENSIVE",
    "ENHANCED", "UNENHANCED", "PORTABLE", "BEDSIDE",
    "PHASE", "DELAYED", "EARLY", "LATE", "DYNAMIC", "STATIC",
    "F18", "F-18", "TC99", "TC-99", "FDG", "PIFLU",
    "WHOLE", "BODY", "WHOLEBODY",
    "FRONTAL", "FRONTAL-ONLY", "LATRL", "OBLIQUE",
    "DIAG", "COMP", "TARGET", "DX", "COMPLETE",
    "SCREEN", "UNILATERAL", "MR", "WO", "CON", "W/CONT",
    "C8906", "CAC", "CALC", "CALCIUM",
    "GUIDE", "GUIDED", "BIOPSY", "BX", "ADDT", "ADDITIONAL",
    "ONLY", "TRANSESOPHAGEAL", "TRANSTHORACIC", "TRANS",
    "R2", "FILM", "DIGITIZED", "STEREO",
    "UNI", "UNI-LFT", "LFT", "RT", "ADDL",
}


BODY_SYNONYMS: dict[str, str] = {
    "ABD": "ABDOMEN", "ABDO": "ABDOMEN", "ABDOMINAL": "ABDOMEN",
    "PELV": "PELVIS", "PELVIC": "PELVIS",
    "HEAD": "BRAIN", "NEURO": "BRAIN", "CRANIAL": "BRAIN", "INTRACRANIAL": "BRAIN",
    "THORAX": "CHEST", "PULMONARY": "CHEST", "LUNGS": "CHEST", "LUNG": "CHEST",
    # NOTE: THORACIC removed — "thoracic spine" != "chest"; handled separately
    "NECK": "NECK", "CERVICAL": "NECK",
    "LUMBAR": "SPINE", "LUMBOSACRAL": "SPINE", "LS": "SPINE", "TL": "SPINE",
    "THORACIC": "SPINE",  # default THORACIC → SPINE unless no SPINE follows
    "MYO": "HEART", "MYOCARDIAL": "HEART", "MYOCARDIUM": "HEART",
    "CARDIAC": "HEART", "CARDIO": "HEART",
    "HEPATIC": "LIVER", "HEP": "LIVER",
    "RENAL": "KIDNEY", "KIDNEYS": "KIDNEY",
    "BREAST": "BREAST", "BREASTS": "BREAST",
    "SKULLTHIGH": "WHOLEBODY", "SKULL": "BRAIN",
    "CORONARY": "HEART", "CARDIO": "HEART",
    "AORTA": "CHEST", "AORTIC": "CHEST",
    "CAROTID": "NECK",
    "FEMORAL": "LEG", "TIBIAL": "LEG",
    "WRIST": "WRIST", "HAND": "HAND", "FINGER": "HAND",
    "SHOULDER": "SHOULDER", "ELBOW": "ELBOW",
    "HIP": "HIP", "KNEE": "KNEE", "ANKLE": "ANKLE", "FOOT": "FOOT",
    "RIBS": "CHEST", "RIB": "CHEST",
    "STERNUM": "CHEST", "CLAVICLE": "CHEST",
    "PELVIC": "PELVIS",
    "ABDOMINAL": "ABDOMEN",
    "PROSTATIC": "PELVIS", "PROSTATE": "PELVIS",
    "UTERUS": "PELVIS", "UTERINE": "PELVIS", "OVARIAN": "PELVIS",
    "SCROTAL": "SCROTUM", "SCROTUM": "SCROTUM",  # scrotal US ≠ pelvic imaging
    "BLADDER": "PELVIS",
    "PANCREAT": "ABDOMEN", "PANCREATIC": "ABDOMEN", "PANCREAS": "ABDOMEN",
    "SPLENIC": "ABDOMEN", "SPLEEN": "ABDOMEN",
    "ADRENAL": "ABDOMEN",
    "GALLBLADDER": "ABDOMEN",
    "PLV": "PELVIS",
    "UROGRAM": "KIDNEY", "UROGRAPHY": "KIDNEY",
    "MUGA": "HEART",
    "ENTEROGRAPHY": "ABDOMEN", "ENTEROCLYSIS": "ABDOMEN",
    "CERV": "NECK",
    "REN": "KIDNEY",  # "CT PLV...REN" = renal
    "LYMPH": "LYMPH",  # lymph node = not a standard body region
    "VENOGRAM": "LEG", "VENOGRAPHY": "LEG",
    "CHOLANGIOGRAPHY": "LIVER", "CHOLANGIOPANCREATOGRAPHY": "LIVER", "MRCP": "LIVER",
    "HEPATOBILIARY": "LIVER",
    "LE": "LEG",
    "EXTREMITY": "LEG",
}

_SYSTEM = (
    "You are an expert radiologist. For a new study, identify which prior exams are relevant for comparison.\n\n"
    "## CORE RULE\n"
    "A prior is RELEVANT when it images the same or overlapping body region with a compatible modality.\n"
    "DEFAULT TO RELEVANT when uncertain — missing a relevant prior is far more harmful than including an extra one.\n\n"
    "## COMPATIBLE MODALITY PAIRS (same body region → mark relevant)\n"
    "- CT ↔ X-ray / plain film / CXR / KUB / CHEST (alone)\n"
    "- MRI ↔ CT\n"
    "- Ultrasound (US, sonogram, sono) ↔ CT or MRI\n"
    "- Nuclear Medicine (NM, SPECT, bone scan, MUGA) ↔ CT or PET\n"
    "- PET/CT ↔ CT or NM (body region may be whole-body — mark relevant to any CT/NM of same region)\n"
    "- Echo (ECHO, TTE, TEE, echocardiogram) ↔ cardiac MRI, cardiac NM, chest CT, chest X-ray\n"
    "- Mammography (MAM, MAMMO, 3D MAMMO, digital screening) ↔ breast US, breast MRI, other mammograms\n"
    "- Same modality ↔ same modality (always relevant if same body region)\n\n"
    "## BODY REGION RULES\n"
    "- Head = Brain = Cranial = Neuro = Intracranial\n"
    "- Chest = Thorax = Thoracic (when no SPINE qualifier) = Pulmonary = Lungs = Ribs = Sternum = Aorta (thoracic)\n"
    "- Heart = Cardiac = Myocardium = MUGA = Coronary — heart is INSIDE chest, so Echo ↔ chest CT/XR is relevant\n"
    "- Abdomen = Abd = Abdominal ≈ Abdomen/Pelvis (overlapping region — MARK RELEVANT)\n"
    "- Pelvis = Pelvic = Bladder = Prostate = Uterus = Ovary = Scrotum\n"
    "- Kidneys = Renal = Urogram — kidneys are in abdomen/pelvis, so renal US ↔ CT abdomen/pelvis is relevant\n"
    "- Spine: cervical/thoracic/lumbar/lumbosacral levels overlap with adjacent levels\n"
    "- Neck = Cervical = Carotid (vessels)\n"
    "- Breast — only relevant to other breast studies (MAM, breast US, breast MRI)\n\n"
    "## SPECIAL CASES\n"
    "- 'CHEST' alone (no other qualifier) = chest X-ray\n"
    "- 'BONE SCAN' / 'BONE SPECT' = Nuclear Medicine whole-body\n"
    "- 'PET/CT skullthigh', 'PET/CT whole body', 'PET/CT F18' = PET whole-body → relevant to CT/NM of any body region\n"
    "- 'ECHO TTE/TEE' = cardiac ultrasound → relevant to chest CT, chest XR, cardiac MRI, cardiac NM\n"
    "- 'ULTRASOUND [site] TARGETED/DIAGNOSTIC/BILATERAL' with no body = likely breast US in breast context → mark relevant to MAM\n"
    "- 'DIGITAL SCREENER', 'STANDARD SCREENING COMBO', 'R2 MAMMOGRAPHY' = mammography\n"
    "- 'CT UROGRAM', 'CT renal colic', 'CT KUB' = CT of kidneys/abdomen/pelvis\n"
    "- 'MRI/CT ANGIO [region]' = vascular study of that region\n\n"
    "## OUTPUT FORMAT\n"
    "You MUST include EVERY prior study_id in your response — no exceptions.\n"
    "Return ONLY valid JSON:\n"
    '{"predictions": [{"study_id": "...", "is_relevant": true}, ...]}'
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

def _is_junk_token(tok: str) -> bool:
    """Filter numeric/short/alphanumeric tokens that are not body parts."""
    if len(tok) <= 1:
        return True
    if tok.isdigit():
        return True
    # tokens like "2V", "3V", "1V", "T2", etc.
    if len(tok) <= 3 and any(c.isdigit() for c in tok):
        return True
    return False


def _parse_modality_and_body(description: str) -> tuple[Optional[str], Optional[str]]:
    # Normalize separators: / ^ _ (and PET-CT specifically)
    text = description.upper()
    text = text.replace("PET-CT", "PET CT").replace("PET/CT", "PET CT")
    text = text.replace("/", " ").replace("^", " ").replace("_", " ").replace("-", " ")

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
            # Handle prefix-attached modality like "NMmyo", "CTchest"
            for alias in ("NM", "CT", "MRI", "US", "MAM"):
                if clean.startswith(alias) and len(clean) > len(alias):
                    modality = MODALITY_ALIASES[alias]
                    mod_token_end = sum(len(t) + 1 for t in tokens[: i + 1])
                    break
            if modality:
                break

    if modality is None:
        tokens = text.split()
        first = tokens[0].rstrip(",;:.") if tokens else ""
        # "CHEST" alone as first meaningful word = XR with body CHEST
        if first == "CHEST":
            modality = "XR"
            mod_token_end = len(tokens[0]) + 1
            return modality, "CHEST"  # body is always CHEST for these
        # "RIBS" or "THORAX" at start = XR of chest region
        elif first in {"RIBS", "RIB"}:
            return "XR", "CHEST"
        elif first == "THORAX":
            return "XR", "CHEST"
        # "DIGITAL SCREENER" / "DIGITAL" mammography descriptions
        elif first == "DIGITAL" or "SCREENER" in text or "MAMMO" in text:
            modality = "MAM"
            mod_token_end = 0
        # "STANDARD SCREENING COMBO", "Dx Bilateral Combo" = mammography combo
        elif ("SCREENING" in text and "COMBO" in text) or ("DX" in text and "COMBO" in text) or ("BILATERAL" in text and "COMBO" in text):
            return "MAM", "BREAST"
        # "BREAST" alone or "BREAST US/MRI" without explicit modality token
        elif first == "BREAST":
            modality = "MAM"  # treat bare BREAST as mammography context
            mod_token_end = len(tokens[0]) + 1
        # "VENOUS IMAGING" or "VENOUS" = venous doppler US
        elif "VENOUS" in text or "DOPPLER" in text:
            modality = "US"
            mod_token_end = 0
        # Standalone or leading body name = plain X-ray of that region
        elif first in {"ABDOMEN", "ABD", "PELVIS", "PELVIC", "SKULL",
                       "HIP", "KNEE", "ANKLE", "SHOULDER", "WRIST", "ELBOW",
                       "LUMBAR", "LUMBOSACRAL", "CERVICAL", "CERVICL",
                       "SPINE", "LUMBAR", "THORACIC"}:
            body_part = BODY_SYNONYMS.get(first, first)
            return "XR", body_part
        # "BONE DENSITY" = DEXA/NM scan of spine/hip
        elif "BONE DENSITY" in text or "DXA" in text or "DEXA" in text:
            return "NM", "SPINE"
        # Mammography screening combos
        elif "TOMOSYNTHESIS" in text or ("SCREENING" in text and "CONVENTIONAL" in text):
            return "MAM", "BREAST"
        # "Standard Screening - Conventional" or "Standard Screening"
        elif first == "STANDARD" and "SCREENING" in text:
            return "MAM", "BREAST"
        # Bare pelvic or single-token body name = XR
        elif len(tokens) == 1 and first in BODY_SYNONYMS:
            return "XR", BODY_SYNONYMS.get(first, first)
        # HEAD at start (e.g. HEAD^SELLA) = brain imaging
        elif first == "HEAD":
            return "MRI", "BRAIN"

    if modality is None:
        return None, None

    # ECHO always maps to HEART
    if modality == "ECHO":
        return modality, "HEART"

    # MAM: track laterality so unilateral LT vs RT can be distinguished
    if modality == "MAM":
        # Detect unilateral side from raw text
        side = None
        for tok in text.split():
            clean = tok.rstrip(",;:./")
            if clean in {"RT", "RIGHT"}:
                side = "RT"
                break
            if clean in {"LT", "LEFT", "LFT"}:
                side = "LT"
                break
        return modality, f"BREAST_{side}" if side else "BREAST"

    # PET whole-body patterns → no specific body region
    if modality == "PET":
        for pattern in _PET_WHOLEBODY_PATTERNS:
            if pattern in text:
                return modality, None

    # Extract body hint from prefix-combined modality token (e.g. "NMmyo" → suffix "MYO" → HEART)
    _suffix_body: Optional[str] = None
    if mod_token_end > 0:
        mod_token = text[:mod_token_end].split()[-1].rstrip(",;:./") if text[:mod_token_end].split() else ""
        for alias in ("NM", "CT", "MRI", "US", "MAM"):
            if mod_token.startswith(alias) and len(mod_token) > len(alias):
                suffix = mod_token[len(alias):]
                _suffix_body = BODY_SYNONYMS.get(suffix)
                break

    # Also check tokens BEFORE the modality for body keywords (e.g. "SCREENING BREAST ULTRASOUND")
    pre_tokens = text[:mod_token_end].split()[:-1]  # everything before modality token
    pre_body: Optional[str] = _suffix_body
    for word in pre_tokens:
        clean = word.rstrip(",;:./")
        if not clean or clean in STOP_WORDS or clean in MODALITY_ALIASES or _is_junk_token(clean):
            continue
        candidate = BODY_SYNONYMS.get(clean, clean)
        if candidate not in STOP_WORDS and candidate not in MODALITY_ALIASES:
            pre_body = candidate
            break

    # Extract body part tokens from the remainder after modality
    remainder = text[mod_token_end:].split()
    body_token: Optional[str] = None
    for word in remainder:
        clean = word.rstrip(",;:./")
        if not clean or clean in STOP_WORDS or clean in MODALITY_ALIASES or _is_junk_token(clean):
            continue
        normalized = BODY_SYNONYMS.get(clean, clean)
        if normalized in STOP_WORDS or normalized in MODALITY_ALIASES:
            continue
        # "WHOLEBODY" means no specific region — leave body as None for LLM
        if normalized == "WHOLEBODY":
            return modality, None
        body_token = normalized
        break

    body = body_token or pre_body

    # Override: if description contains renal-specific keywords, prefer KIDNEY over PELVIS
    if body == "PELVIS":
        upper = text  # already uppercased
        if any(kw in upper for kw in (" REN", "RENAL", "KIDNEY", "UROGRAM", "COLIC")):
            body = "KIDNEY"

    _CANONICAL_BODIES = {
        "BREAST", "HEART", "BRAIN", "CHEST", "ABDOMEN", "PELVIS",
        "SPINE", "NECK", "LIVER", "KIDNEY", "SHOULDER", "ELBOW",
        "WRIST", "HAND", "HIP", "KNEE", "ANKLE", "FOOT", "LEG",
    }
    # Last-resort: scan full text for known body keywords if still no body
    if body is None:
        for word in text.split():
            clean = word.rstrip(",;:./")
            candidate = BODY_SYNONYMS.get(clean)
            if candidate and candidate not in {"WHOLEBODY"} and candidate not in STOP_WORDS:
                body = candidate
                break
            if clean in _CANONICAL_BODIES:
                body = clean
                break

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

    def _breast_match(b1: Optional[str], b2: Optional[str]) -> bool:
        """BREAST_RT and BREAST_LT are not relevant to each other."""
        if b1 is None or b2 is None:
            return False
        if b1 == b2:
            return True
        # One or both are lateralized — only match if same side or at least one is bilateral
        if b1.startswith("BREAST") and b2.startswith("BREAST"):
            sides1 = b1.replace("BREAST", "").strip("_") or "BI"
            sides2 = b2.replace("BREAST", "").strip("_") or "BI"
            return "BI" in {sides1, sides2} or sides1 == sides2
        return False

    if not cur_body or not pri_body:
        if "MAM" in {cur_mod, pri_mod} and cur_mod != pri_mod:
            mam_body = cur_body if cur_mod == "MAM" else pri_body
            other_body = pri_body if cur_mod == "MAM" else cur_body
            # Allow if MAM side is BREAST and other has no body (targeted breast US/MRI)
            return (mam_body or "").startswith("BREAST") and other_body is None
        # PET with no body = whole-body PET → relevant to any compatible CT/NM
        if "PET" in {cur_mod, pri_mod} and mod_pair in CROSS_MODALITY:
            pet_body = cur_body if cur_mod == "PET" else pri_body
            if pet_body is None:
                return True
        return same_mod

    # Both bodies known — MAM cross-modality only when both are breast region
    if "MAM" in {cur_mod, pri_mod} and cur_mod != pri_mod:
        return _breast_match(cur_body, pri_body)

    # MAM same-modality: respect laterality
    if same_mod and cur_mod == "MAM":
        return _breast_match(cur_body, pri_body)

    if cur_body == pri_body:
        return True

    # Selected clinical body overlaps (anatomical containment or clinical co-imaging)
    _OVERLAPS: dict[str, frozenset] = {
        "KIDNEY": frozenset({"ABDOMEN"}),         # kidneys in abdomen, NOT pelvis proper
        "ABDOMEN": frozenset({"KIDNEY", "PELVIS", "LIVER"}),
        "PELVIS": frozenset({"ABDOMEN", "BLADDER"}),  # no KIDNEY — scrotum/pelvis ≠ kidney
        "BLADDER": frozenset({"PELVIS"}),
        "LIVER": frozenset({"ABDOMEN"}),
    }
    return pri_body in _OVERLAPS.get(cur_body, frozenset())


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
        max_tokens=8192,
        system=[
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={"effort": "high"},
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
