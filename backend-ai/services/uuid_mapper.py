"""Map UUID user_id to integer user_id for model inference.

Uses the uuid_mapping.json artifact produced by load_db_datasets_from_db().
If the mapping file does not exist, falls back to treating the UUID as-is
(for environments where the DB already uses integer IDs).
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Simple in-memory cache for UUID → int mapping
_uuid_to_int_cache: dict[str, int] = {}
_int_to_uuid_cache: dict[int, str] = {}


def _load_mapping() -> dict[str, int]:
    """Load UUID→int mapping from artifacts, if present."""
    mapping_path = Path(__file__).resolve().parents[2] / "data" / "artifacts" / "uuid_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_mapping = data.get("uuid_to_int", {})
            return {str(key).upper(): int(value) for key, value in raw_mapping.items()}
        except Exception:
            pass
    return {}


def _get_rev_mapping() -> dict[str, int]:
    """Load int→UUID reverse mapping from artifacts, if present."""
    mapping_path = Path(__file__).resolve().parents[2] / "data" / "artifacts" / "uuid_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_mapping = data.get("int_to_uuid", {})
            return {str(key): str(value).upper() for key, value in raw_mapping.items()}
        except Exception:
            pass
    return {}


def get_mentee_integer_id(user_uuid: str) -> Optional[int]:
    """Convert UUID user_id to integer user_id for model inference.

    Args:
        user_uuid: The UUID string from frontend/API.

    Returns:
        Integer user_id if mapping exists, None otherwise.
    """
    # Check cache first
    if user_uuid in _uuid_to_int_cache:
        return _uuid_to_int_cache[user_uuid]

    # Try to load from mapping file
    mapping = _load_mapping()
    user_id_int = mapping.get(str(user_uuid).upper())
    if user_id_int is not None:
        _uuid_to_int_cache[str(user_uuid)] = int(user_id_int)
        return int(user_id_int)

    # Fallback: if the value itself is numeric, use it directly
    try:
        val = int(user_uuid)
        _uuid_to_int_cache[str(user_uuid)] = val
        return val
    except (ValueError, TypeError):
        pass

    logger.warning("User UUID not found in mapping and is not numeric: %s", user_uuid)
    return None


def get_uuid_from_integer(user_int: int | str) -> Optional[str]:
    """Reverse-map an integer user_id back to UUID string.

    Args:
        user_int: The integer user_id from model output.

    Returns:
        UUID string if mapping exists, input as string otherwise.
    """
    if isinstance(user_int, int):
        user_int = str(user_int)
    if user_int in _int_to_uuid_cache:
        return _int_to_uuid_cache[user_int]

    rev_mapping = _get_rev_mapping()
    uuid_val = rev_mapping.get(str(user_int))
    if uuid_val is not None:
        _int_to_uuid_cache[str(user_int)] = uuid_val
        return uuid_val

    # If no reverse mapping, return the input as-is (it might already be a UUID)
    return user_int


def clear_cache() -> None:
    """Clear both UUID→int and int→UUID mapping caches."""
    global _uuid_to_int_cache, _int_to_uuid_cache
    _uuid_to_int_cache.clear()
    _int_to_uuid_cache.clear()
    logger.info("UUID mapping caches cleared (forward + reverse)")
