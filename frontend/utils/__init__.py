"""
frontend.utils — Dashboard support modules (Person 4)
==============================================================================

    api_client.py   HTTP calls to Person 3's FastAPI backend
    data_utils.py   Loading and filtering records for the tables
    charts.py       Plotly figures (gauge, pie, bar, distribution)
    pdf_report.py   ReportLab PDF generation

WHY THE REPO ROOT IS PUT ON sys.path HERE
    Streamlit is launched as `streamlit run frontend/app.py`, which places
    `frontend/` on sys.path — not the repository root. Without the bootstrap
    below, `from frontend.utils import ...` and, more importantly,
    `from backend.data_loader import ...` both fail.

    That backend import is deliberate and is the reason this matters. The
    dashboard reads Person 2's CSV through the *same* `data_loader` the API
    uses, so the gauge on screen and the figures the LLM reasons over come
    from one calculation. A separate frontend copy of that logic would drift
    the first time a threshold changed, and the dashboard would confidently
    display numbers the backend disagreed with.
==============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

#: <repo_root>/frontend/utils/__init__.py -> up three levels is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    # Position 0 so the project's own `backend` and `models` packages win over
    # any similarly-named third-party distribution.
    sys.path.insert(0, str(REPO_ROOT))

__all__ = ["REPO_ROOT"]
