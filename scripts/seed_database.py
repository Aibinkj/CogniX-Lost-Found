"""Create the schema and populate it with realistic found items.

    python scripts/seed_database.py            # drop and re-create everything
    python scripts/seed_database.py --keep     # only add missing rows

The catalogue deliberately contains near-misses - a black Bose headset and a
navy Sony headset alongside the black Sony pair - so that semantic similarity
alone is not enough and the confidence signals have to do real work.

The help knowledge base (desks and FAQs) is synced by slug on every run, with
or without --keep, so editing the lists below and re-running updates it in place.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from app.config import configure_logging, settings  # noqa: E402
from app.database.connection import get_engine, session_scope  # noqa: E402
from app.database.models import Base, FoundItem, HelpArticle, User  # noqa: E402
from app.database.vector_support import current_backend  # noqa: E402
from app.rag.embeddings import embed_many, get_embedder  # noqa: E402

configure_logging()
logger = logging.getLogger("seed")

NOW = datetime.now(timezone.utc)


def hours_ago(hours: float) -> datetime:
    return NOW - timedelta(hours=hours)


# (category, brand, color, description, hidden_features, location, found_time)
FOUND_ITEMS: list[tuple[str, str | None, str, str, str, str, datetime]] = [
    # --- the demo target ---------------------------------------------------
    (
        "headphones", "Sony", "black",
        "Black Sony wireless over-ear headphones with a folding headband",
        "scratch on left earcup; small sticker inside case",
        "Library Security Desk", hours_ago(20),
    ),
    # --- deliberate near-misses for the same query -------------------------
    (
        "headphones", "Bose", "black",
        "Black Bose noise-cancelling over-ear headphones in a hard case",
        "owner's initials M.K. written on the headband; frayed cable",
        "Library Security Desk", hours_ago(26),
    ),
    (
        "headphones", "Sony", "navy",
        "Navy blue Sony wired headphones with a coiled cable",
        "left hinge is cracked; blue tape on the cable",
        "Sports Centre Reception", hours_ago(60),
    ),
    (
        "earbuds", "Sony", "black",
        "Black Sony wireless earbuds in a matte charging case",
        "chip on the corner of the charging case lid; one earbud tip is missing",
        "Library Security Desk", hours_ago(18),
    ),
    (
        "headphones", "JBL", "red",
        "Red JBL on-ear headphones with a fabric headband",
        "band logo is peeling; a hair tie is wrapped around the cable",
        "Student Union Desk", hours_ago(40),
    ),
    # --- earbuds -----------------------------------------------------------
    (
        "earbuds", "Apple", "white",
        "White Apple AirPods in a charging case",
        "engraved with the word Milo on the lid; deep scratch on the hinge",
        "Cafeteria Counter", hours_ago(8),
    ),
    (
        "earbuds", "Samsung", "white",
        "White Samsung Galaxy Buds in a round case",
        "cracked hinge on the case; a green paint mark on the underside",
        "Lecture Hall B Reception", hours_ago(30),
    ),
    # --- phones ------------------------------------------------------------
    (
        "phone", "Apple", "black",
        "Black iPhone with a clear plastic case",
        "photo of a golden retriever on the lock screen; hairline crack in the top-left corner",
        "Cafeteria Counter", hours_ago(10),
    ),
    (
        "phone", "Samsung", "silver",
        "Silver Samsung Galaxy phone with a cracked screen protector",
        "a bus ticket tucked inside the case; sticker of a blue whale on the back",
        "Main Reception", hours_ago(34),
    ),
    (
        "phone", "Google", "green",
        "Green Google Pixel phone, no case",
        "small dent on the bottom edge; wallpaper is a photo of mountains",
        "Bus Stop Office", hours_ago(52),
    ),
    # --- wallets -----------------------------------------------------------
    (
        "wallet", None, "brown",
        "Brown leather bifold wallet",
        "library card in the name of A. Rahman; folded receipt from a bookshop inside",
        "Library Security Desk", hours_ago(22),
    ),
    (
        "wallet", None, "black",
        "Black fabric card holder with a metal clip",
        "three transit cards inside; a torn cinema stub in the back pocket",
        "Gym Reception", hours_ago(45),
    ),
    # --- backpacks ---------------------------------------------------------
    (
        "backpack", "Herschel", "navy",
        "Navy Herschel backpack with tan leather straps",
        "a small enamel cat pin on the front pocket; chemistry textbook inside",
        "Lecture Hall A Reception", hours_ago(28),
    ),
    (
        "backpack", "JanSport", "black",
        "Black JanSport backpack with a broken zip pull",
        "a red keyring shaped like a bicycle; name tag written in marker inside the flap",
        "Sports Centre Reception", hours_ago(70),
    ),
    # --- laptops -----------------------------------------------------------
    (
        "laptop", "Apple", "silver",
        "Silver Apple MacBook Air in a grey sleeve",
        "three stickers on the lid including a rocket; dead pixel near the top right",
        "Computer Lab 2", hours_ago(14),
    ),
    (
        "laptop", "Dell", "black",
        "Black Dell laptop with a worn trackpad",
        "asset tag ending 4471 underneath; missing rubber foot at the back left",
        "Main Reception", hours_ago(58),
    ),
    # --- watches -----------------------------------------------------------
    (
        "watch", "Casio", "silver",
        "Silver Casio digital watch with a metal bracelet",
        "engraving on the back reads 2019; one link removed from the strap",
        "Swimming Pool Reception", hours_ago(36),
    ),
    (
        "watch", "Fitbit", "black",
        "Black Fitbit fitness tracker with a silicone strap",
        "the strap is cut shorter than standard; scuff across the screen",
        "Gym Reception", hours_ago(12),
    ),
    # --- water bottles -----------------------------------------------------
    (
        "water_bottle", "Hydro Flask", "teal",
        "Teal Hydro Flask insulated bottle, one litre",
        "dent near the base; sticker of a mountain range on the side",
        "Sports Centre Reception", hours_ago(16),
    ),
    (
        "water_bottle", "Contigo", "black",
        "Black Contigo travel mug with a flip lid",
        "name scratched into the base; the lid seal is bright orange",
        "Cafeteria Counter", hours_ago(6),
    ),
    # --- books -------------------------------------------------------------
    (
        "book", None, "blue",
        "Blue hardback organic chemistry textbook, third edition",
        "handwritten notes in the margins of chapter four; a boarding pass used as a bookmark",
        "Library Security Desk", hours_ago(48),
    ),
    (
        "book", None, "black",
        "Black A5 hardback notebook with an elastic band",
        "first page has a phone number and no name; a pressed leaf between the last pages",
        "Study Room 3", hours_ago(24),
    ),
    # --- chargers ----------------------------------------------------------
    (
        "charger", "Anker", "white",
        "White Anker power bank with a short USB-C cable",
        "masking tape label reading PHYS on the side; one port is taped over",
        "Computer Lab 2", hours_ago(32),
    ),
    (
        "charger", "Apple", "white",
        "White Apple laptop charger with a wound cable",
        "a strip of blue insulating tape near the plug; the cable is frayed at the brick",
        "Lecture Hall B Reception", hours_ago(44),
    ),
    # --- misc --------------------------------------------------------------
    (
        "keys", None, "silver",
        "Bunch of three keys on a plain metal ring",
        "a small bottle-opener charm; one key has a green rubber cover",
        "Main Reception", hours_ago(9),
    ),
    (
        "glasses", None, "black",
        "Black rectangular glasses in a soft grey pouch",
        "left arm has been repaired with a mismatched screw; optician's card inside the pouch",
        "Library Security Desk", hours_ago(38),
    ),
    (
        "umbrella", None, "navy",
        "Navy compact folding umbrella",
        "the wrist strap is knotted where it snapped; two broken ribs",
        "Student Union Desk", hours_ago(50),
    ),
]

USERS: list[tuple[str, str]] = [
    ("Alex Rahman", "alex.rahman@example.edu"),
    ("Priya Nair", "priya.nair@example.edu"),
    ("Sam Okafor", "sam.okafor@example.edu"),
]

# ---------------------------------------------------------------------------
# Help knowledge base.
#
# SAMPLE DATA. Every hour, extension, email address and policy below is a
# placeholder for the demo, and every row is stored with is_sample=True so the
# UI labels it. Replace with the real desk details before real use.
# ---------------------------------------------------------------------------

# (desk_location, title, where, hours, phone, email). desk_location matches the
# `location` values used for found items, so pickups can quote their desk.
SERVICE_DESKS: list[tuple[str, str, str, str, str, str]] = [
    (
        "Main Reception", "Lost & Found Office (Main Reception)",
        "Ground floor of the Administration Block, beside the main entrance. This is the central "
        "Lost & Found Office: it reviews escalated cases and holds items not collected from a local "
        "desk within 7 days.",
        "Mon-Sat 8:00 AM - 8:00 PM", "Campus ext. 2100", "lostandfound@example.edu",
    ),
    (
        "Library Security Desk", "Library Security Desk",
        "Library ground floor, just inside the main doors.",
        "Every day 7:00 AM - 11:00 PM", "Campus ext. 2110", "library.security@example.edu",
    ),
    (
        "Study Room 3", "Library Security Desk (for Study Room 3)",
        "Items left in the library study rooms are held at the Library Security Desk, library "
        "ground floor.",
        "Every day 7:00 AM - 11:00 PM", "Campus ext. 2110", "library.security@example.edu",
    ),
    (
        "Student Union Desk", "Student Union Desk",
        "Student Union building, first-floor information counter.",
        "Mon-Fri 9:00 AM - 6:00 PM", "Campus ext. 2120", "union.desk@example.edu",
    ),
    (
        "Sports Centre Reception", "Sports Centre Reception",
        "Main entrance of the Sports Centre.",
        "Every day 6:00 AM - 10:00 PM", "Campus ext. 2130", "sports.reception@example.edu",
    ),
    (
        "Gym Reception", "Gym Reception",
        "Entrance of the gym, inside the Sports Centre.",
        "Every day 6:00 AM - 10:00 PM", "Campus ext. 2131", "gym.reception@example.edu",
    ),
    (
        "Swimming Pool Reception", "Swimming Pool Reception",
        "Pool entrance, beside the changing rooms.",
        "Every day 6:30 AM - 9:00 PM", "Campus ext. 2132", "pool.reception@example.edu",
    ),
    (
        "Lecture Hall A Reception", "Lecture Hall A Reception",
        "Foyer of Lecture Hall A.",
        "Mon-Fri 8:00 AM - 6:00 PM", "Campus ext. 2140", "lecturehall.a@example.edu",
    ),
    (
        "Lecture Hall B Reception", "Lecture Hall B Reception",
        "Foyer of Lecture Hall B.",
        "Mon-Fri 8:00 AM - 6:00 PM", "Campus ext. 2141", "lecturehall.b@example.edu",
    ),
    (
        "Cafeteria Counter", "Cafeteria Counter",
        "Items handed in at the cafeteria are kept behind the cashier counter.",
        "Mon-Sat 7:30 AM - 9:00 PM", "Campus ext. 2150", "cafeteria@example.edu",
    ),
    (
        "Computer Lab 2", "IT Helpdesk (for Computer Lab 2)",
        "Items left in Computer Lab 2 are held at the IT Helpdesk next to the lab entrance.",
        "Mon-Fri 8:30 AM - 7:00 PM", "Campus ext. 2160", "it.helpdesk@example.edu",
    ),
    (
        "Bus Stop Office", "Bus Stop Office",
        "Transport office at the main campus bus stop.",
        "Mon-Sat 7:00 AM - 7:00 PM", "Campus ext. 2170", "transport@example.edu",
    ),
]

_CENTRAL = SERVICE_DESKS[0]

# (slug, title, body)
HELP_FAQS: list[tuple[str, str, str]] = [
    (
        "faq-contact-staff", "How to contact Lost & Found staff",
        f"You can reach Lost & Found staff at the {_CENTRAL[1]}: {_CENTRAL[4]}, {_CENTRAL[5]}, "
        f"open {_CENTRAL[3]}. Each collection desk can also be contacted directly.",
    ),
    (
        "faq-collect-item", "Where and how to collect your item",
        "After your ownership is verified you get a pickup reference such as LF1024. Go to the desk "
        "named in your confirmation during its opening hours, bring photo ID and quote the reference. "
        "Items are reserved for you for 7 days.",
    ),
    (
        "faq-escalated-case", "What happens when a case is escalated to staff",
        "When the assistant cannot confidently match or verify an item, it opens a staff review case "
        "with a reference such as ESC500. Staff review cases within one working day and contact you. "
        "To follow up sooner, contact the Lost & Found Office and quote your reference.",
    ),
    (
        "faq-found-item", "Found something? Where to hand in an item you found",
        "Please hand found items to the nearest collection desk or to the Lost & Found Office at Main "
        "Reception. Staff record where and when it was found, and any distinguishing marks are kept "
        "private so they can be used to check ownership.",
    ),
    (
        "faq-retention", "How long found items are kept",
        "Items stay at the desk where they were handed in for 7 days, then move to the Lost & Found "
        "Office at Main Reception. Unclaimed items are kept for 30 days in total before being donated "
        "or disposed of.",
    ),
    (
        "faq-collect-for-someone", "Collecting an item on someone else's behalf",
        "Items are released only to the verified owner in person with photo ID. If you cannot collect "
        "it yourself, contact the Lost & Found Office to arrange an authorised collection.",
    ),
    (
        "faq-ownership-check", "Why the assistant asks for a distinctive feature",
        "To stop items going to the wrong person, you must describe a detail only the owner would "
        "know that is not in the public description. Staff never reveal the details on file, so "
        "they cannot be repeated back.",
    ),
    (
        "faq-not-found-yet", "If your item has not been handed in yet",
        "Your report is saved even when nothing matches yet, and staff check new items against open "
        "reports. Check back later, or contact the Lost & Found Office with your reference.",
    ),
    (
        "faq-what-assistant-does", "What this assistant can do",
        "This assistant searches items handed in at campus desks, scores how confident a match is, "
        "verifies ownership before releasing anything, creates pickup requests, answers questions "
        "about desks and collection, and checks references such as LF1024 or ESC500.",
    ),
]


def _slugify(text_value: str) -> str:
    return "desk-" + "-".join("".join(ch if ch.isalnum() else " " for ch in text_value.lower()).split())


def help_articles() -> list[dict[str, object]]:
    """The knowledge base as rows. Desk bodies are written out so they embed well."""
    rows: list[dict[str, object]] = []
    for location, title, where, hours, phone, email in SERVICE_DESKS:
        rows.append(
            {
                "slug": _slugify(location),
                "kind": "desk",
                "title": title,
                "body": (
                    f"{title}. Where: {where} Opening hours: {hours}. "
                    f"Phone: {phone}. Email: {email}."
                ),
                "desk_location": location,
                "hours": hours,
                "phone": phone,
                "email": email,
            }
        )
    for slug, title, body in HELP_FAQS:
        rows.append({"slug": slug, "kind": "faq", "title": title, "body": body, "desk_location": None,
                     "hours": None, "phone": None, "email": None})
    return rows


def sync_help_articles(session) -> int:
    """Insert or update every article by slug and drop ones no longer listed."""
    rows = help_articles()
    existing = {article.slug: article for article in session.scalars(select(HelpArticle)).all()}
    wanted = {str(row["slug"]) for row in rows}

    for slug, article in existing.items():
        if slug not in wanted:
            session.delete(article)

    articles: list[HelpArticle] = []
    for row in rows:
        article = existing.get(str(row["slug"])) or HelpArticle(slug=row["slug"])
        for field, value in row.items():
            setattr(article, field, value)
        article.is_sample = True
        articles.append(article)

    vectors = embed_many([article.embedding_text() for article in articles])
    for article, vector in zip(articles, vectors):
        article.embedding = vector
    session.add_all(articles)
    return len(articles)


def reset_schema() -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Schema re-created.")


def ensure_schema() -> None:
    Base.metadata.create_all(get_engine())


def seed(keep: bool = False) -> None:
    get_engine()  # triggers vector-backend detection before any DDL

    if keep:
        ensure_schema()
    else:
        reset_schema()

    embedder = get_embedder()
    logger.info("Embeddings: %s (%d dims).", embedder.name, embedder.dim)
    logger.info("Vector backend: %s.", current_backend())

    with session_scope() as session:
        synced = sync_help_articles(session)
    logger.info("Synced %d help articles (sample desk and FAQ content).", synced)

    with session_scope() as session:
        existing = session.scalar(select(func.count()).select_from(FoundItem)) or 0
        if keep and existing:
            logger.info("Keeping %d existing found items.", existing)
            return

        for name, email in USERS:
            if not session.scalar(select(User).where(User.email == email)):
                session.add(User(name=name, email=email))

        items = [
            FoundItem(
                category=category,
                brand=brand,
                color=color,
                description=description,
                hidden_features=hidden,
                location=location,
                found_time=found_time,
                status="available",
            )
            for category, brand, color, description, hidden, location, found_time in FOUND_ITEMS
        ]

        vectors = embed_many([item.embedding_text() for item in items])
        for item, vector in zip(items, vectors):
            item.embedding = vector

        session.add_all(items)

    logger.info("Seeded %d found items and %d users.", len(FOUND_ITEMS), len(USERS))


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the Lost & Found database.")
    parser.add_argument(
        "--keep", action="store_true", help="Keep existing data instead of dropping the schema."
    )
    args = parser.parse_args()

    logger.info("Database: %s", settings.database_url.rsplit("@", 1)[-1])
    try:
        seed(keep=args.keep)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly CLI error
        logger.error("Seeding failed: %s", exc)
        logger.error(
            "Check DATABASE_URL in .env and that the server is running "
            "(`docker compose up -d` starts the bundled pgvector instance)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
