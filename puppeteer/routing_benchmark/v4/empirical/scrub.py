from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SENSITIVE_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:ghp|github_pat|glpat)-?[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*)"
        r"([^\s,;\]\[{}\"']+)"
    ),
)
WINDOWS_QUOTED_PATH = re.compile(r"([\"'])([A-Za-z]:[\\/][^\"'\r\n]+)\1")
WINDOWS_BARE_PATH = re.compile(r"(?<![\w<])([A-Za-z]:[\\/][^\s\"'<>|]+)")
POSIX_PRIVATE_PATH = re.compile(r"(?<![\w<])/(?:home|Users|tmp|var/tmp)/[^\s\"'<>]+")
UNC_PATH = re.compile(r"(?<![\w<])\\\\[^\\\s]+\\[^\s\"'<>]+")


def _env_secret_values(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    values: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if not any(part in name.upper() for part in SENSITIVE_NAME_PARTS):
            continue
        value = value.strip().strip("\"").strip("'")
        if len(value) >= 8:
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _normalized_variants(path: Path) -> tuple[str, ...]:
    raw = str(path.resolve())
    return tuple(dict.fromkeys((raw, raw.replace("\\", "/"), raw.replace("/", "\\"))))


class SensitiveDataScrubber:
    """Deterministically redact local identity and credentials while retaining task evidence."""

    def __init__(
        self,
        *,
        project_root: Path,
        env_file: Path | None = None,
        extra_secret_values: Iterable[str] = (),
    ) -> None:
        self.project_variants = _normalized_variants(project_root)
        self.home_variants = _normalized_variants(Path.home())
        self.usernames = {
            value
            for value in (os.environ.get("USERNAME"), os.environ.get("USER"), Path.home().name)
            if value and len(value) >= 3
        }
        self.secret_values = sorted(
            {
                *(_env_secret_values(env_file)),
                *(value for value in extra_secret_values if len(value) >= 8),
            },
            key=len,
            reverse=True,
        )

    @staticmethod
    def _scrub_url(value: str) -> str:
        if not value.lower().startswith(("http://", "https://")):
            return value
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        sensitive = {"key", "api_key", "apikey", "token", "access_token", "secret", "password"}
        query = [
            (name, "<REDACTED_SECRET>" if name.lower() in sensitive else item)
            for name, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), parsed.fragment))

    def scrub_text(self, value: str) -> str:
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                structured = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                structured = None
            if isinstance(structured, (dict, list)):
                return json.dumps(
                    self.scrub(structured), ensure_ascii=False, sort_keys=True
                )
        result = value
        for secret in self.secret_values:
            result = result.replace(secret, "<REDACTED_SECRET>")
        for root in self.project_variants:
            result = result.replace(root, "<PROJECT_ROOT>")
        for root in self.home_variants:
            result = result.replace(root, "<USER_HOME>")
        for username in self.usernames:
            result = re.sub(re.escape(username), "<LOCAL_USER>", result, flags=re.IGNORECASE)
        for pattern in SECRET_PATTERNS:
            if pattern.groups:
                result = pattern.sub(r"\1<REDACTED_SECRET>", result)
            else:
                result = pattern.sub("<REDACTED_SECRET>", result)
        result = WINDOWS_QUOTED_PATH.sub(
            lambda match: f'{match.group(1)}<LOCAL_PATH>{match.group(1)}', result
        )
        result = WINDOWS_BARE_PATH.sub("<LOCAL_PATH>", result)
        result = POSIX_PRIVATE_PATH.sub("<LOCAL_PATH>", result)
        result = UNC_PATH.sub("<LOCAL_PATH>", result)
        if result.lower().startswith(("http://", "https://")):
            result = self._scrub_url(result)
        return result

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Mapping):
            return {str(key): self.scrub(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.scrub(child) for child in value]
        if isinstance(value, tuple):
            return [self.scrub(child) for child in value]
        return deepcopy(value)


def residual_sensitive_matches(value: Any) -> list[str]:
    """Return categories only; never return the matched secret text."""

    findings: list[str] = []
    checks = {
        # The boundary avoids treating escaped source snippets such as
        # ``try:\\n`` or ``value:\\n`` as Windows drive paths.
        "windows_absolute_path": re.compile(r"(?<![A-Za-z])[A-Z]:[\\/]"),
        "posix_private_path": re.compile(r"/(?:home|Users|tmp|var/tmp)/"),
        "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    }
    def strings(item: Any) -> Iterable[str]:
        if isinstance(item, str):
            yield item
        elif isinstance(item, Mapping):
            for key, child in item.items():
                yield str(key)
                yield from strings(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                yield from strings(child)

    values = tuple(strings(value))
    for name, pattern in checks.items():
        if any(pattern.search(text) for text in values):
            findings.append(name)
    return findings
