"""Compatibility wrapper for the shared cleaner module.

Episode 1 originally stored the cleaner implementation here. As the series grows,
shared workflow code now lives in `shared.ai_workflow` so every episode can reuse
one source of truth instead of copying files around.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.ai_workflow.cleaner import *  # noqa: F401,F403

