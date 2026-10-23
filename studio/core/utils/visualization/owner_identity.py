"""Read the current account-owner name from explicit persisted facts."""
import re


_NAME_FACTS = (
    re.compile(r"^(?:用户|使用者)(?:现在|目前)?(?:的(?:新)?名字(?:是|为|叫)|(?:改名为|更名为|改叫|叫|名叫|自称))\s*[「『\"“]?([^，,。.!！?？；;（(\n」』\"”]+)"),
    re.compile(r"^(?:the\s+)?user(?:'s\s+(?:new\s+)?name\s+is|\s+is\s+(?:now\s+)?(?:named|called)|\s+(?:has\s+)?changed\s+(?:their\s+)?name\s+to)\s+[\"“]?([^,.!?;\n\"”]+)", re.I),
)
_HISTORICAL = re.compile(r"曾用|曾经|以前|过去|原来|formerly|previous(?:ly)?|used to|old name", re.I)


def current_owner_name(entries):
    """Return the newest explicit current name, excluding other people and history.

    This is a display projection of stored account-owner facts. It never binds
    voiceprints, changes historical records, or infers identity from dialogue.
    """
    ordered = sorted(entries, key=lambda item: str(item.get("updated_at") or item.get("date") or ""),
                     reverse=True)
    for item in ordered:
        text = (item.get("text") or "").strip()
        if (item.get("role") == "assistant" or _HISTORICAL.search(text)
                or any(c in text for c in "?？")
                or re.search(r"不是|并非|可能|是否|\b(?:not|isn't|might|maybe)\b", text, re.I)):
            continue
        for pattern in _NAME_FACTS:
            match = pattern.match(text)
            if match:
                name = match.group(1).strip()
                if (0 < len(name) <= 60 and not any(c in name for c in "什谁哪啥吗呢么?？")
                        and all(c.isalpha() or c in " '-·" for c in name)):
                    return name
    return ""
