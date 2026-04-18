"""Ultralite Code Assistant Engine

Core modules are canonically defined in densanon-core (densanon.core.*).
This engine/ directory contains code-specific extensions.
Shared modules import from densanon-core.
"""

__version__ = "0.5.0"

import sys
from pathlib import Path

_CORE_ROOT = Path(__file__).parent.parent.parent / "densanon-core"
if _CORE_ROOT.exists() and str(_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CORE_ROOT))
