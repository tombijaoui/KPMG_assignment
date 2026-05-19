from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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


class QAToolCallFunction(BaseModel):
    name: str
    arguments: str


class QAToolCall(BaseModel):
    id: str
    type: str = "function"
    function: QAToolCallFunction


class QAChatMessage(BaseModel):
    """OpenAI-compatible chat message for Q&A history (includes tool turns)."""

    role: str = Field(..., description="user, assistant, or tool")
    content: str = ""
    tool_calls: list[QAToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_message(self) -> QAChatMessage:
        if self.role not in ("user", "assistant", "tool"):
            raise ValueError(f"unsupported role: {self.role}")
        if self.role == "user" and not self.content.strip():
            raise ValueError("user message content is required")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return self


class QAToolUsage(BaseModel):
    """One tool invocation from the latest /qa turn, in call order."""

    order: int = Field(..., ge=1, description="1-based position in the tool call sequence")
    name: str
    arguments: str
    tool_call_id: str


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


class QARequest(BaseModel):
    message: str = Field(..., min_length=1, description="Latest user question")
    user_profile: UserProfile = Field(..., description="Confirmed member profile")
    messages: list[QAChatMessage] = Field(
        default_factory=list,
        description="Prior Q&A turns (user, assistant, tool), oldest first",
    )
    profile_confirmed: bool = Field(
        default=False,
        description="Must be true when the member has approved their profile",
    )


class QAResponse(BaseModel):
    reply: str
    turn_messages: list[QAChatMessage] = Field(
        default_factory=list,
        description="Full turn to append to client history (user, tools, final assistant)",
    )
    tool_calls: list[QAToolUsage] = Field(
        default_factory=list,
        description="Tools invoked this turn, in call order (empty if none)",
    )
