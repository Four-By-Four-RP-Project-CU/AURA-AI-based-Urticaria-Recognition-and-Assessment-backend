import io
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

try:
    from gridfs import GridFSBucket
    from pymongo import MongoClient
except Exception:  # pragma: no cover - optional dependency
    GridFSBucket = None
    MongoClient = None


logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "").strip()
MONGO_DB = os.getenv("MONGODB_DB", "aura_ai").strip()
RISK_COLLECTION = os.getenv("MONGODB_RISK_COLLECTION", "risk_results").strip()
PRESCRIPTION_COLLECTION = os.getenv("MONGODB_PRESCRIPTION_COLLECTION", "prescription_results").strip()
RISK_BUCKET = os.getenv("MONGODB_RISK_BUCKET", "risk_assets").strip()
PRESCRIPTION_BUCKET = os.getenv("MONGODB_PRESCRIPTION_BUCKET", "prescription_assets").strip()

_client: Optional["MongoClient"] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_case_id() -> str:
    return f"AURA-{uuid.uuid4().hex[:8].upper()}"


def mongo_enabled() -> bool:
    return bool(MONGO_URI and MongoClient is not None and GridFSBucket is not None)


def _get_client() -> Optional["MongoClient"]:
    global _client
    if not mongo_enabled():
        return None
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client


def _serialize_pil(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _store_binary(
    *,
    bucket_name: str,
    case_id: str,
    module: str,
    kind: str,
    filename: str,
    content_type: str,
    payload: bytes,
) -> Optional[dict[str, Any]]:
    client = _get_client()
    if client is None:
        return None

    db = client[MONGO_DB]
    bucket = GridFSBucket(db, bucket_name=bucket_name)
    file_id = bucket.upload_from_stream(
        filename,
        payload,
        metadata={
            "case_id": case_id,
            "module": module,
            "kind": kind,
            "content_type": content_type,
            "saved_at": utc_now_iso(),
        },
    )
    return {
        "file_id": str(file_id),
        "filename": filename,
        "content_type": content_type,
        "kind": kind,
        "bucket": bucket_name,
    }


def _store_uploaded_files(
    *,
    bucket_name: str,
    case_id: str,
    module: str,
    kind: str,
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for item in files:
        payload = item.get("content")
        if not payload:
            continue
        ref = _store_binary(
            bucket_name=bucket_name,
            case_id=case_id,
            module=module,
            kind=kind,
            filename=item.get("filename") or f"{kind}.bin",
            content_type=item.get("content_type") or "application/octet-stream",
            payload=payload,
        )
        if ref:
            refs.append(ref)
    return refs


def _store_generated_images(
    *,
    bucket_name: str,
    case_id: str,
    module: str,
    images: dict[str, Image.Image],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for kind, pil_image in images.items():
        if pil_image is None:
            continue
        ref = _store_binary(
            bucket_name=bucket_name,
            case_id=case_id,
            module=module,
            kind=kind,
            filename=f"{case_id}_{kind}.png",
            content_type="image/png",
            payload=_serialize_pil(pil_image, "PNG"),
        )
        if ref:
            refs.append(ref)
    return refs


def _upsert_document(collection_name: str, case_id: str, document: dict[str, Any]) -> bool:
    client = _get_client()
    if client is None:
        return False
    db = client[MONGO_DB]
    db[collection_name].update_one({"case_id": case_id}, {"$set": document}, upsert=True)
    return True


def save_risk_result(
    *,
    case_id: str,
    patient_name: str = "",
    endpoint: str,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    uploaded_files: Optional[list[dict[str, Any]]] = None,
) -> bool:
    if not mongo_enabled():
        return False
    try:
        asset_refs = _store_uploaded_files(
            bucket_name=RISK_BUCKET,
            case_id=case_id,
            module="risk",
            kind="lab_report",
            files=uploaded_files or [],
        )
        now = utc_now_iso()
        document = {
            "case_id": case_id,
            "patient_name": patient_name or None,
            "module": "risk",
            "endpoint": endpoint,
            "request_payload": request_payload,
            "result_payload": result_payload,
            "asset_refs": asset_refs,
            "updated_at": now,
            "created_at": now,
        }
        return _upsert_document(RISK_COLLECTION, case_id, document)
    except Exception as exc:  # pragma: no cover - operational fallback
        logger.warning("Mongo risk persistence failed for case %s: %s", case_id, exc)
        return False


def save_prescription_result(
    *,
    case_id: str,
    patient_name: str = "",
    endpoint: str,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    uploaded_files: Optional[list[dict[str, Any]]] = None,
    generated_images: Optional[dict[str, Image.Image]] = None,
) -> bool:
    if not mongo_enabled():
        return False
    try:
        asset_refs = _store_uploaded_files(
            bucket_name=PRESCRIPTION_BUCKET,
            case_id=case_id,
            module="prescription",
            kind="input_asset",
            files=uploaded_files or [],
        )
        if generated_images:
            asset_refs.extend(
                _store_generated_images(
                    bucket_name=PRESCRIPTION_BUCKET,
                    case_id=case_id,
                    module="prescription",
                    images=generated_images,
                )
            )
        now = utc_now_iso()
        document = {
            "case_id": case_id,
            "patient_name": patient_name or None,
            "module": "prescription",
            "endpoint": endpoint,
            "request_payload": request_payload,
            "result_payload": result_payload,
            "asset_refs": asset_refs,
            "updated_at": now,
            "created_at": now,
        }
        return _upsert_document(PRESCRIPTION_COLLECTION, case_id, document)
    except Exception as exc:  # pragma: no cover - operational fallback
        logger.warning("Mongo prescription persistence failed for case %s: %s", case_id, exc)
        return False


def get_case_bundle(case_id: str) -> Optional[dict[str, Any]]:
    client = _get_client()
    if client is None:
        return None
    db = client[MONGO_DB]
    risk_doc = db[RISK_COLLECTION].find_one({"case_id": case_id}, {"_id": 0})
    prescription_doc = db[PRESCRIPTION_COLLECTION].find_one({"case_id": case_id}, {"_id": 0})
    if not risk_doc and not prescription_doc:
        return None
    return {
        "case_id": case_id,
        "mongo_db": MONGO_DB,
        "risk_result": risk_doc,
        "prescription_result": prescription_doc,
    }
