"""Redact NAS paths before any data leaves the VPN boundary."""
from __future__ import annotations

import re
import unicodedata

REDACTED = "[percorso NAS riservato]"
_CONFUSABLE = str.maketrans({
    "⁄": "/", "∕": "/", "／": "/", "м": "m", "М": "m", "ｍ": "m", "Ｍ": "m",
    "п": "n", "П": "n", "ｎ": "n", "Ｎ": "n", "т": "t", "Т": "t", "ｔ": "t", "Ｔ": "t",
    "а": "a", "А": "a", "ａ": "a", "Ａ": "a", "ѕ": "s", "Ѕ": "s", "ｓ": "s", "Ｓ": "s",
})
_PREFIX = re.compile(r"[/⁄∕／]\s*[mмМｍＭ]\s*[nпПｎＮ]\s*[tтТｔＴ]\s*[/⁄∕／]\s*[nпПｎＮ]\s*[aаАａＡ]\s*[sѕЅｓＳ]\s*(?=[/⁄∕／])", re.I)
_LOCAL = re.compile(
    # A NAS path with a file extension may contain spaces ("Folder with spaces/x.pdf");
    # a bare directory path may not, or it would swallow the surrounding prose.
    r"/[mM][nN][tT]/[nN][aA][sS]/(?:[^\x00\"'<>|\]\)]|[\r\n])+?\.[A-Za-z0-9]{1,12}(?=$|[\s\"'<>|\]\),;:!?])"
    r"|/[mM][nN][tT]/[nN][aA][sS](?:/[^\x00\s\"'<>|\]\),;:!?]+)*")
_UNC = re.compile(r"//(?:192\.168\.52\.37|nas\.therness\.com)(?:/|\\\\)(?:[^\x00\r\n\"'<>|\]\)]|[\r\n])+?(?=$|[\s\"'<>|\]\),;:!?])", re.I)


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    value = _PREFIX.sub(lambda m: re.sub(r"\s+", "", m.group()).translate(_CONFUSABLE), value)
    return re.sub(r"/mnt/nas/[ \t]*\r?\n[ \t]*(?=\S)", "/mnt/nas/", value, flags=re.I)


def redact(value: object) -> str:
    """Redact all local/UNC NAS references incl. spaced, split, homoglyph prefixes."""
    out = _normalise(str(value))
    out = _LOCAL.sub(REDACTED, out)
    return _UNC.sub(REDACTED, out)


def contains_nas_path(value: object) -> bool:
    return REDACTED in redact(value)
