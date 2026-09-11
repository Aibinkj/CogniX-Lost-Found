"""Turn a free-text lost-property report into structured item details.

Uses the configured LLM when one is available and falls back to a deterministic
keyword/regex parser otherwise. The active path is reported in
``ItemDetails.extracted_by`` so the trace panel never implies a model ran when
none did.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agent.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE
from app.llm import LLMError, get_llm

logger = logging.getLogger(__name__)

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "headphones": ("headphone", "headphones", "headset", "over-ear", "cans"),
    "earbuds": ("earbud", "earbuds", "airpod", "airpods", "earphone", "earphones", "buds"),
    "phone": ("phone", "iphone", "smartphone", "mobile", "android", "pixel", "galaxy"),
    "wallet": ("wallet", "purse", "billfold", "cardholder"),
    "backpack": ("backpack", "rucksack", "bag", "satchel", "knapsack"),
    "laptop": ("laptop", "macbook", "notebook computer", "chromebook", "thinkpad"),
    "watch": ("watch", "smartwatch", "fitbit", "garmin"),
    "water_bottle": ("bottle", "flask", "thermos", "tumbler", "hydro"),
    "book": ("book", "textbook", "novel", "notebook", "journal"),
    "charger": ("charger", "cable", "power bank", "adapter", "powerbank", "charging"),
    "keys": ("key", "keys", "keychain", "keyring", "fob"),
    "umbrella": ("umbrella", "brolly"),
    "glasses": ("glasses", "spectacles", "sunglasses", "eyeglasses"),
    "tablet": ("tablet", "ipad", "kindle"),
    "jacket": ("jacket", "coat", "hoodie", "sweater", "cardigan"),
}

BRANDS: tuple[str, ...] = (
    "sony", "bose", "apple", "samsung", "jbl", "sennheiser", "anker", "dell",
    "hp", "lenovo", "asus", "acer", "microsoft", "google", "fitbit", "garmin",
    "casio", "seiko", "fossil", "hydro flask", "contigo", "chilly's", "nike",
    "adidas", "north face", "herschel", "jansport", "beats", "skullcandy",
    "logitech", "xiaomi", "oneplus", "huawei", "kindle", "amazon", "belkin",
)

COLORS: tuple[str, ...] = (
    "black", "white", "silver", "grey", "gray", "blue", "navy", "red", "green",
    "yellow", "orange", "purple", "pink", "brown", "beige", "gold", "rose gold",
    "teal", "maroon", "charcoal", "cream", "transparent", "clear",
)

LOCATION_HINTS: tuple[str, ...] = (
    "library", "cafeteria", "canteen", "gym", "sports centre", "sports center",
    "lecture hall", "lecture theatre", "classroom", "lab", "laboratory",
    "student union", "union", "bus stop", "bus", "train", "station", "parking",
    "car park", "dorm", "dormitory", "residence hall", "cafe", "coffee shop",
    "bookstore", "book shop", "auditorium", "reception", "front desk",
    "security desk", "study room", "computer lab", "main hall", "courtyard",
    "swimming pool", "pool", "field", "stadium", "office", "corridor",
)

_LOCATION_PHRASE_RE = re.compile(
    r"\b(?:near|at|in|inside|by|outside|around|next to|beside)\s+"
    r"(?:the\s+)?([a-z][a-z' -]{2,40})",
    re.IGNORECASE,
)

_CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.IGNORECASE)
_H24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

_STOPWORDS_AFTER_LOCATION = (
    " yesterday", " today", " this ", " last ", " around", " about", " at ",
    " on ", " sometime", " earlier", " late", " morning", " afternoon",
    " evening", " night", " and ",
)


@dataclass
class ItemDetails:
    """Structured view of what the user lost."""

    category: str | None = None
    brand: str | None = None
    color: str | None = None
    location: str | None = None
    time_phrase: str | None = None
    lost_time: datetime | None = None
    description: str = ""
    extracted_by: str = "rules"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lost_time"] = self.lost_time.isoformat() if self.lost_time else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ItemDetails | None":
        """Rebuild from :meth:`to_dict`, e.g. state carried between turns."""
        if not payload:
            return None
        lost_time = payload.get("lost_time")
        if isinstance(lost_time, str):
            try:
                lost_time = datetime.fromisoformat(lost_time)
            except ValueError:
                lost_time = None
        return cls(
            category=payload.get("category"),
            brand=payload.get("brand"),
            color=payload.get("color"),
            location=payload.get("location"),
            time_phrase=payload.get("time_phrase"),
            lost_time=lost_time,
            description=payload.get("description", ""),
            extracted_by=payload.get("extracted_by", "rules"),
            warnings=list(payload.get("warnings") or []),
        )

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.category:
            missing.append("category")
        if not self.location:
            missing.append("location")
        if not self.color and not self.brand:
            missing.append("colour or brand")
        return missing

    def search_text(self) -> str:
        """The string handed to the embedder."""
        parts = [self.color, self.brand, self.category.replace("_", " ") if self.category else None]
        if self.description:
            parts.append(self.description)
        if self.location:
            parts.append(f"lost near {self.location}")
        seen: set[str] = set()
        unique: list[str] = []
        for part in parts:
            if part and part.lower() not in seen:
                seen.add(part.lower())
                unique.append(part)
        return " ".join(unique).strip()

    def known_field_count(self) -> int:
        return sum(1 for value in (self.category, self.brand, self.color, self.location) if value)

    def merge(self, other: "ItemDetails") -> "ItemDetails":
        """Overlay newly-learned fields from a follow-up message."""
        return ItemDetails(
            category=other.category or self.category,
            brand=other.brand or self.brand,
            color=other.color or self.color,
            location=other.location or self.location,
            time_phrase=other.time_phrase or self.time_phrase,
            lost_time=other.lost_time or self.lost_time,
            description=(f"{self.description} {other.description}".strip() or other.description),
            extracted_by=other.extracted_by,
            warnings=other.warnings,
        )


def _clean_location(raw: str) -> str:
    text = raw.strip().lower()
    for stopword in _STOPWORDS_AFTER_LOCATION:
        index = text.find(stopword)
        if index > 0:
            text = text[:index]
    return re.sub(r"[\s,.;:]+$", "", text).strip()


def find_location(message: str) -> str | None:
    lowered = message.lower()
    for match in _LOCATION_PHRASE_RE.finditer(message):
        candidate = _clean_location(match.group(1))
        if len(candidate) >= 3:
            return candidate
    for hint in sorted(LOCATION_HINTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(hint)}\b", lowered):
            return hint
    return None


def parse_time_phrase(message: str, now: datetime | None = None) -> tuple[str | None, datetime | None]:
    """Resolve a relative time expression to an absolute timestamp."""
    now = now or datetime.now(timezone.utc)
    lowered = message.lower()

    day_offset: int | None = None
    day_phrase = ""
    if "day before yesterday" in lowered:
        day_offset, day_phrase = 2, "day before yesterday"
    elif "yesterday" in lowered:
        day_offset, day_phrase = 1, "yesterday"
    elif "last night" in lowered:
        day_offset, day_phrase = 1, "last night"
    elif "this morning" in lowered:
        day_offset, day_phrase = 0, "this morning"
    elif "this afternoon" in lowered:
        day_offset, day_phrase = 0, "this afternoon"
    elif "this evening" in lowered or "tonight" in lowered:
        day_offset, day_phrase = 0, "this evening"
    elif "today" in lowered:
        day_offset, day_phrase = 0, "today"
    elif match := re.search(r"\b(\d+)\s+days?\s+ago\b", lowered):
        day_offset, day_phrase = int(match.group(1)), match.group(0)

    hour: int | None = None
    minute = 0
    clock_phrase = ""
    if clock := _CLOCK_RE.search(message):
        hour = int(clock.group(1)) % 12
        minute = int(clock.group(2) or 0)
        if clock.group(3).lower() == "pm":
            hour += 12
        clock_phrase = clock.group(0)
    elif h24 := _H24_RE.search(message):
        hour, minute = int(h24.group(1)), int(h24.group(2))
        clock_phrase = h24.group(0)
    elif "morning" in lowered:
        hour, clock_phrase = 9, "morning"
    elif "afternoon" in lowered:
        hour, clock_phrase = 15, "afternoon"
    elif "evening" in lowered or "night" in lowered:
        hour, clock_phrase = 20, "evening"

    if day_offset is None and hour is None:
        return None, None

    base = now - timedelta(days=day_offset or 0)
    resolved = base.replace(
        hour=hour if hour is not None else 12, minute=minute, second=0, microsecond=0
    )
    if resolved > now:
        resolved -= timedelta(days=1)

    phrase = " ".join(part for part in (day_phrase, clock_phrase) if part).strip()
    return (phrase or None), resolved


def extract_with_rules(message: str, now: datetime | None = None) -> ItemDetails:
    """Deterministic keyword + regex extraction. No model involved."""
    lowered = message.lower()

    category = None
    best_position = len(lowered) + 1
    for name, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            match = re.search(rf"\b{re.escape(keyword)}\b", lowered)
            if match and match.start() < best_position:
                category, best_position = name, match.start()

    brand = next(
        (b for b in sorted(BRANDS, key=len, reverse=True) if re.search(rf"\b{re.escape(b)}\b", lowered)),
        None,
    )
    color = next(
        (c for c in sorted(COLORS, key=len, reverse=True) if re.search(rf"\b{re.escape(c)}\b", lowered)),
        None,
    )

    time_phrase, lost_time = parse_time_phrase(message, now=now)

    return ItemDetails(
        category=category,
        brand=brand.title() if brand else None,
        color=color,
        location=find_location(message),
        time_phrase=time_phrase,
        lost_time=lost_time,
        description=message.strip(),
        extracted_by="rules",
    )


def _coerce(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown", "n/a", "unclear"}:
        return None
    return text


def extract_item_details(message: str, now: datetime | None = None) -> ItemDetails:
    """Extract with the LLM when configured; always fall back to the rules."""
    baseline = extract_with_rules(message, now=now)

    client = get_llm()
    if client is None:
        return baseline

    try:
        payload = client.complete_json(
            EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE.format(message=message), max_tokens=400
        )
    except (LLMError, KeyError, ValueError) as exc:
        logger.warning("LLM extraction failed (%s); using rule-based extraction.", exc)
        baseline.warnings.append(f"LLM extraction failed, used rules: {exc}")
        return baseline

    time_phrase = _coerce(payload.get("time_phrase"))
    lost_time = baseline.lost_time
    if time_phrase and lost_time is None:
        _, lost_time = parse_time_phrase(time_phrase, now=now)

    category = _coerce(payload.get("category"))
    if category:
        category = category.lower().replace(" ", "_")
        if category not in CATEGORY_KEYWORDS and category != "other":
            category = baseline.category

    return ItemDetails(
        category=category or baseline.category,
        brand=_coerce(payload.get("brand")) or baseline.brand,
        color=(_coerce(payload.get("color")) or baseline.color or "").lower() or None,
        location=(_coerce(payload.get("location")) or baseline.location or "").lower() or None,
        time_phrase=time_phrase or baseline.time_phrase,
        lost_time=lost_time,
        description=_coerce(payload.get("description")) or baseline.description,
        extracted_by=f"llm:{client.provider}:{client.model}",
    )
