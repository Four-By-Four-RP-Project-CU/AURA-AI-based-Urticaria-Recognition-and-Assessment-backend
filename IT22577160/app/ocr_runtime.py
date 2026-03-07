import re
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import pytesseract

# Point pytesseract to the Tesseract executable on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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