"""ORM models for the Lost & Found prototype."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings
from app.database.vector_support import EmbeddingVector


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lost_items: Mapped[list["LostItem"]] = relationship(back_populates="user")


class FoundItem(Base):
    """An item handed in at a desk.

    `hidden_features` is ownership evidence. It is never returned by search and
    never rendered in the UI; only `app.tools.verification` reads it.
    """

    __tablename__ = "found_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category: Mapped[str] = mapped_column(String(60), nullable=False, index=True)
    brand: Mapped[str | None] = mapped_column(String(80))
    color: Mapped[str | None] = mapped_column(String(60))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    hidden_features: Mapped[str] = mapped_column(Text, nullable=False, default="")
    location: Mapped[str] = mapped_column(String(160), nullable=False)
    found_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="available", index=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    claims: Mapped[list["Claim"]] = relationship(back_populates="found_item")
    pickup_requests: Mapped[list["PickupRequest"]] = relationship(back_populates="found_item")

    def public_dict(self) -> dict[str, Any]:
        """Everything a user is allowed to see. Deliberately omits hidden_features."""
        return {
            "id": self.id,
            "category": self.category,
            "brand": self.brand,
            "color": self.color,
            "description": self.description,
            "location": self.location,
            "found_time": self.found_time.isoformat() if self.found_time else None,
            "status": self.status,
        }

    def embedding_text(self) -> str:
        parts = [self.color, self.brand, self.category, self.description, f"found at {self.location}"]
        return " ".join(part for part in parts if part)


class LostItem(Base):
    __tablename__ = "lost_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    category: Mapped[str | None] = mapped_column(String(60), index=True)
    brand: Mapped[str | None] = mapped_column(String(80))
    color: Mapped[str | None] = mapped_column(String(60))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str | None] = mapped_column(String(160))
    lost_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_query: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="searching", index=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User | None] = relationship(back_populates="lost_items")
    claims: Mapped[list["Claim"]] = relationship(back_populates="lost_item")


class Claim(Base):
    """One user's attempt to claim one found item, with its verification outcome."""

    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lost_item_id: Mapped[int | None] = mapped_column(ForeignKey("lost_items.id", ondelete="CASCADE"))
    found_item_id: Mapped[int] = mapped_column(
        ForeignKey("found_items.id", ondelete="CASCADE"), nullable=False
    )
    confidence_score: Mapped[float | None] = mapped_column(Float)
    confidence_level: Mapped[str | None] = mapped_column(String(20))
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    verification_score: Mapped[float | None] = mapped_column(Float)
    user_answer: Mapped[str | None] = mapped_column(Text)
    reasons: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    lost_item: Mapped[LostItem | None] = relationship(back_populates="claims")
    found_item: Mapped[FoundItem] = relationship(back_populates="claims")


class PickupRequest(Base):
    __tablename__ = "pickup_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reference: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    claim_id: Mapped[int | None] = mapped_column(ForeignKey("claims.id", ondelete="SET NULL"))
    found_item_id: Mapped[int] = mapped_column(
        ForeignKey("found_items.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    location: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    found_item: Mapped[FoundItem] = relationship(back_populates="pickup_requests")


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    reference: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    user_request: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_ids: Mapped[list[int] | None] = mapped_column(JSONB)
    confidence_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    verification_result: Mapped[str | None] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class HelpArticle(Base):
    """Help-desk knowledge: one collection desk or one FAQ entry, embedded for RAG.

    Desk rows carry structured contact fields so escalation and pickup messages
    can quote them directly. The shipped rows are sample content (`is_sample`)
    and must be replaced with the real desk details before real use.
    """

    __tablename__ = "help_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False, default="faq")
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # For desks: the `found_items.location` / `pickup_requests.location` it serves.
    desk_location: Mapped[str | None] = mapped_column(String(160), index=True)
    hours: Mapped[str | None] = mapped_column(String(160))
    phone: Mapped[str | None] = mapped_column(String(60))
    email: Mapped[str | None] = mapped_column(String(120))
    is_sample: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(settings.embedding_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "slug": self.slug,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "desk_location": self.desk_location,
            "hours": self.hours,
            "phone": self.phone,
            "email": self.email,
            "is_sample": self.is_sample,
        }

    def embedding_text(self) -> str:
        return f"{self.title}. {self.body}"


class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel: Mapped[str] = mapped_column(String(30), nullable=False, default="in_app")
    recipient: Mapped[str] = mapped_column(String(200), nullable=False)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


Index("ix_found_items_status_category", FoundItem.status, FoundItem.category)
