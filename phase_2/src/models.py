from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Collected health-fund user profile (stateless; sent on every request)."""

    first_name: str | None = None
    last_name: str | None = None
    id_number: str | None = None
    gender: str | None = None
    age: int | None = None
    hmo: str | None = None
    hmo_card_number: str | None = None
    insurance_tier: str | None = None


class ProfilePatch(BaseModel):
    """Fields extracted from the latest user message (omit unknown values)."""

    first_name: str | None = None
    last_name: str | None = None
    id_number: str | None = None
    gender: str | None = None
    age: int | None = None
    hmo: str | None = None
    hmo_card_number: str | None = None
    insurance_tier: str | None = None


class CollectLLMOutput(BaseModel):
    """Structured response from the collect-phase LLM."""

    reply: str
    profile_patch: ProfilePatch = Field(default_factory=ProfilePatch)
    profile_confirmed: bool = False


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., min_length=1)


class CollectInfoRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user message")
    user_profile: UserProfile = Field(default_factory=UserProfile)
    recent_messages: list[ChatMessage] = Field(
        default_factory=list,
        description="Last few chat turns for confirmation and correction context",
    )


class CollectInfoResponse(BaseModel):
    reply: str
    user_profile: UserProfile
    profile_confirmed: bool = False
    profile_valid: bool = False
    ready_for_qa: bool = False
