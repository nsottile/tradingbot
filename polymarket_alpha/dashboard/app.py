"""Compatibility entrypoint for the new Streamlit UI package."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# region agent log
_DEBUG_LOG = Path(__file__).resolve().parents[2] / ".cursor" / "debug-349ba6.log"


def _agent_log(message: str, data: dict, hypothesis_id: str, run_id: str = "deploy-fix") -> None:
    payload = {
        "sessionId": "349ba6",
        "timestamp": int(time.time() * 1000),
        "location": "polymarket_alpha/dashboard/app.py",
        "message": message,
        "data": data,
        "hypothesisId": hypothesis_id,
        "runId": run_id,
    }
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


# endregion

_repo_root = Path(__file__).resolve().parents[2]
_agent_log(
    "pre_bootstrap",
    {
        "cwd": str(Path.cwd()),
        "__file__": str(Path(__file__).resolve()),
        "repo_root": str(_repo_root),
        "sys_path_head": sys.path[:10],
        "pkg_init_exists": (_repo_root / "polymarket_alpha" / "__init__.py").exists(),
        "repo_in_path": str(_repo_root) in sys.path,
    },
    "A",
)

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_agent_log(
    "post_bootstrap",
    {"sys_path_head": sys.path[:5], "repo_in_path": str(_repo_root) in sys.path},
    "B",
)

try:
    from polymarket_alpha.ui.app import render_app
except (ModuleNotFoundError, ImportError) as exc:
    _agent_log(
        "import_failed",
        {"error": str(exc), "exc_type": type(exc).__name__, "name": getattr(exc, "name", None)},
        "C",
    )
    raise

_agent_log("import_ok", {"module": "polymarket_alpha.ui.app", "has_render_app": callable(render_app)}, "C")

render_app()
