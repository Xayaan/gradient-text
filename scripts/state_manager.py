import json
import os
import tempfile
from pathlib import Path

try:
    import redis
except Exception:
    redis = None

STATE_KEY = "state"
STATE_BACKEND_ENV = "TEXT_MINER_STATE_BACKEND"
STATE_FILE_PATH_ENV = "TEXT_MINER_STATE_FILE"


def _get_state_file_path() -> Path:
    configured_path = os.getenv(STATE_FILE_PATH_ENV)
    if configured_path:
        return Path(configured_path)
    return Path(__file__).resolve().parent / ".trainer_state.json"


def _load_state_from_file() -> dict:
    path = _get_state_file_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state_to_file(state: dict) -> None:
    path = _get_state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as f:
        json.dump(state, f)
        temp_name = f.name
    os.replace(temp_name, path)


def _get_redis_client():
    if redis is None:
        return None
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD", None)
    db = int(os.getenv("REDIS_DB", 0))
    client = redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)
    client.ping()
    return client


def _should_use_redis() -> bool:
    backend = os.getenv(STATE_BACKEND_ENV, "auto").strip().lower()
    if backend == "redis":
        return True
    if backend == "file":
        return False
    return redis is not None


def get_state() -> dict:
    if _should_use_redis():
        try:
            client = _get_redis_client()
            if client is not None:
                value = client.get(STATE_KEY)
                if value is None:
                    return {}
                return json.loads(value)
        except Exception:
            pass
    return _load_state_from_file()


def set_state(state: dict) -> None:
    if _should_use_redis():
        try:
            client = _get_redis_client()
            if client is not None:
                client.set(STATE_KEY, json.dumps(state))
                return
        except Exception:
            pass
    _save_state_to_file(state)


def test():
    state = get_state()
    print(json.dumps(state, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    test()
