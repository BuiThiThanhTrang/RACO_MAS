from __future__ import annotations

import html
import json
from typing import Any


def pretty_json(value: Any) -> str:
    """Return stable, readable JSON without losing non-ASCII text."""

    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def wrapped_json_html(value: Any) -> str:
    """Render JSON as safe HTML that wraps long values instead of scrolling sideways."""

    escaped = html.escape(pretty_json(value), quote=True)
    return f'<pre class="rb-json-wrap">{escaped}</pre>'


def safe_text(value: Any) -> str:
    """Escape dataset text before placing it in a custom HTML fragment."""

    return html.escape(str(value or ""), quote=True)
