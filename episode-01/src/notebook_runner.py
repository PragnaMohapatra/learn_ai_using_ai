"""Compatibility wrapper for the shared notebook runner module.

The actual implementation now lives in `shared.ai_workflow.notebook_runner` so
future episodes can reuse the same execution logic without copying files.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.ai_workflow.notebook_runner import *  # noqa: F401,F403

