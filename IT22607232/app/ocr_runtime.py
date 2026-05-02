import re
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import pytesseract

# Use an explicit tesseract path only when one is configured or on Windows.
_tesseract_cmd = (
    os.getenv("TESSERACT_CMD")
    or os.getenv("PYTESSERACT_CMD")
)
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd
elif os.name == "nt":
    _windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_windows_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _windows_tesseract

def normalize_text(t: str) -> str:
    t = t.replace("μ", "u").replace("µ", "u").replace("×", "x")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 8)
    thr = cv2.bitwise_not(thr)
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.bitwise_not(thr)
    return thr

def ocr_bytes(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    proc = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(proc, config="--oem 3 --psm 6")
    return normalize_text(text)

def _clean_num(s: str) -> Optional[float]:
    s = s.replace(",", "").strip()
    s = re.sub(r"^[<>]\s*", "", s)
    try:
        return float(s)
    except:
        return None

def _norm_unit(u: str) -> str:
    return re.sub(r"\s+", "", (u or "").lower())

def vitd_to_ng_ml(v: float, unit: str) -> float:
    u = _norm_unit(unit)
    if u in ["nmol/l", "nmoll"]:
        return v / 2.5
    return v

def extract_vitd(text: str) -> Optional[float]:
    t = text.lower()
    pat = r"vitamin\s*d.*?(?:25\s*hydroxy.*?)?([<>]?\s*\d+(?:\.\d+)?)\s*([a-zA-Z/]+)?"
    m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    v = _clean_num(m.group(1))
    u = (m.group(2) or "").strip()
    if v is None:
        return None
    return vitd_to_ng_ml(v, u)

def find_lab(text: str, patterns: List[str]) -> List[Tuple[str, float, str]]:
    hits = []
    for nm in patterns:
        # (?:\([^)]*\))? skips an optional parenthetical abbreviation such as
        # "IMMUNOGLOBULIN E (IgE LEVEL)  455.60" or "FREE THYROXINE (F.T4)  0.890"
        pattern = rf"({nm})\s*(?:\([^)]*\))?\s*[:=]?\s*([<>]?\s*\d+(?:\.\d+)?)\s*([a-zA-Z0-9/\^\-\.\s]+)?"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            val = _clean_num(m.group(2))
            unit = (m.group(3) or "").strip()
            if val is not None:
                hits.append((m.group(1), val, unit))
    return hits

LAB_SYNONYMS = {
    "CRP":  [r"\bcrp\b", r"c[\-\s]?reactive\s*protein", r"c\.?\s*reactive"],
    "FT4":  [r"\bft[\s\-]?4\b", r"free\s*t[\s\-]?4", r"free\s*thyroxine",
             r"t4[\s,]*free", r"thyroxine[\s,]*free", r"f\.?\s*t\.?\s*4"],
    "IgE":  [r"\big[\s\-]?e\b", r"immunoglobulin[\s\-]?e", r"total[\s\-]?ig[\s\-]?e",
             r"ige[\s\-]?total", r"ig\.?\s*e\b"],
    "Age":  [r"\bage\b"],
    # kept for completeness / future use
    "ESR":     [r"\besr\b"],
    "WBC":     [r"\bwbc\b", r"white\s*blood\s*cell"],
    "HB":      [r"haemoglobin", r"hemoglobin", r"\bhb\b"],
    "PLT":     [r"platelet\s*count", r"\bplt\b", r"\bplatelets\b"],
    "EOS_ABS": [r"eosinophils#?", r"eos#"],
}

def extract_labs_from_text(text: str) -> Dict[str, Any]:
    out = {
        # Model lab features
        "CRP":  None,
        "FT4":  None,
        "IgE":  None,
        "VitD": None,
        "Age":  None,
        # Extra values from report (informational)
        "ESR":             None,
        "WBC":             None,
        "Hb":              None,
        "Platelets":       None,
        "Eosinophils_abs": None,
        "flags": {"missing": []},
    }

    # VitD (unit-aware)
    vit = extract_vitd(text)
    if vit is not None:
        out["VitD"] = vit
    else:
        out["flags"]["missing"].append("VitD")

    # CRP
    crp = find_lab(text, LAB_SYNONYMS["CRP"])
    out["CRP"] = crp[-1][1] if crp else None
    if not crp: out["flags"]["missing"].append("CRP")

    # FT4
    ft4 = find_lab(text, LAB_SYNONYMS["FT4"])
    out["FT4"] = ft4[-1][1] if ft4 else None
    if not ft4: out["flags"]["missing"].append("FT4")

    # IgE
    ige = find_lab(text, LAB_SYNONYMS["IgE"])
    out["IgE"] = ige[-1][1] if ige else None
    if not ige: out["flags"]["missing"].append("IgE")

    # Age
    age = find_lab(text, LAB_SYNONYMS["Age"])
    out["Age"] = age[-1][1] if age else None
    if not age: out["flags"]["missing"].append("Age")

    # Extra / informational
    esr = find_lab(text, LAB_SYNONYMS["ESR"])
    out["ESR"] = esr[-1][1] if esr else None

    wbc = find_lab(text, LAB_SYNONYMS["WBC"])
    out["WBC"] = wbc[-1][1] if wbc else None

    hb = find_lab(text, LAB_SYNONYMS["HB"])
    out["Hb"] = hb[-1][1] if hb else None

    plt = find_lab(text, LAB_SYNONYMS["PLT"])
    out["Platelets"] = plt[-1][1] if plt else None

    eos = find_lab(text, LAB_SYNONYMS["EOS_ABS"])
    out["Eosinophils_abs"] = eos[-1][1] if eos else None

    return out

def extract_labs_from_images(image_bytes_list: List[bytes]) -> Dict[str, Any]:
    full_text = "\n".join([ocr_bytes(b) for b in image_bytes_list if b])
    return extract_labs_from_text(full_text)


# ── Document type detection ──────────────────────────────────────────────────

_DOC_RULES: List[Tuple[List[str], str]] = [
    (["vitamin d", "25 hydroxy"], "LAB"),
    (["c-reactive", "crp"], "LAB"),
    (["ige", "immunoglobulin e"], "LAB"),
    (["ft4", "free t4", "free thyroxine"], "LAB"),
    (["urine full report", "ufr"], "LAB"),
    (["wbc", "haematology", "hematology", "platelet"], "LAB"),
    (["c/o", "plan", "noct", "bd", "od", "tab", "cap", "mg", "ml"], "PRESCRIPTION"),
]


def guess_doc_type(text: str) -> str:
    tl = text.lower()
    if "esr" in tl and any(u in tl for u in ("mm/hr", "mmhr")):
        return "LAB"
    for keywords, label in _DOC_RULES:
        if any(k in tl for k in keywords):
            return label
    return "LAB"  # default: treat as lab text for safety


# ── BERT text helpers ────────────────────────────────────────────────────────

def make_lab_summary(text: str) -> str:
    """Keep only lines that mention relevant lab markers (for investigations_raw)."""
    keys = [
        "crp", "c-reactive", "ige", "vitamin d", "25 hydroxy", "ft4", "free t4",
        "esr", "wbc", "hb", "haemoglobin", "hemoglobin", "platelet", "plt",
        "eosinophils", "neutrophils", "lymphocytes",
    ]
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out: List[str] = []
    seen: set = set()
    for ln in lines:
        lw = ln.lower()
        if any(k in lw for k in keys):
            if not re.search(r"(tel|fax|www|hospital|diagnostics|confidential|page)", lw):
                if ln not in seen:
                    out.append(ln)
                    seen.add(ln)
    return "\n".join(out[:120]).strip()


def make_symptoms_text(text: str) -> str:
    """Extract drug/symptom lines from prescription OCR output."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    meds: List[str] = []
    for ln in lines:
        lw = ln.lower()
        if any(x in lw for x in ["prof", "faculty", "university", "reg no",
                                   "immunology", "allergy", "hospital", "diagnostic"]):
            continue
        if re.search(r"\d+\s*(mg|mcg|ug|ml)", lw):
            meds.append(ln)
            continue
        if re.search(r"\b(puff|bd|od|noct|tid|qid)\b", lw):
            meds.append(ln)
            continue
        if len(ln) < 35 and re.search(r"[a-zA-Z]{4,}", ln):
            meds.append(ln)
    seen2: set = set()
    unique = []
    for m in meds:
        if m not in seen2:
            unique.append(m)
            seen2.add(m)
    return "\n".join(unique[:20]).strip()


# ── PDF support (requires pymupdf) ────────────────────────────────────────────

def _pdf_to_images(path: str) -> List[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("Install pymupdf for PDF support: pip install pymupdf")
    doc = fitz.open(path)
    images = []
    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        images.append(arr)
    doc.close()
    return images


# ── Main entry: process one uploaded file (bytes) ─────────────────────────────

def process_upload(filename: str, file_bytes: bytes) -> Dict[str, List[str]]:
    """
    Process one uploaded file (image or PDF).
    Returns {"lab_texts": [...], "rx_texts": [...]}
    """
    ext = os.path.splitext(filename.lower())[1]
    lab_texts: List[str] = []
    rx_texts:  List[str] = []

    if ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            images = _pdf_to_images(tmp_path)
            for img in images:
                _, buf = cv2.imencode(".png", img)
                raw = ocr_bytes(buf.tobytes())
                (rx_texts if guess_doc_type(raw) == "PRESCRIPTION" else lab_texts).append(raw)
        finally:
            os.unlink(tmp_path)
    else:
        raw = ocr_bytes(file_bytes)
        (rx_texts if guess_doc_type(raw) == "PRESCRIPTION" else lab_texts).append(raw)

    return {"lab_texts": lab_texts, "rx_texts": rx_texts}


def build_ocr_result(lab_texts: List[str], rx_texts: List[str]) -> Dict[str, Any]:
    """
    Combine OCR outputs into a dict ready for the prediction endpoint.
    Returns:
      labs_extracted    : {CRP, FT4, IgE, VitD, ...} (float or None)
      investigations_raw: cleaned lab summary text for BERT
      symptoms_raw      : cleaned prescription/symptom text for BERT
      missing_fields    : lab keys not found in images
    """
    combined_lab  = "\n".join(lab_texts)
    combined_rx   = "\n".join(rx_texts)

    labs = extract_labs_from_text(combined_lab)
    missing = labs.pop("flags", {}).get("missing", [])

    return {
        "labs_extracted":    dict(labs),
        "investigations_raw": make_lab_summary(combined_lab),
        "symptoms_raw":       make_symptoms_text(combined_rx),
        "missing_fields":     missing,
    }
