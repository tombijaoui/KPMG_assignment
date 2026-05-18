from __future__ import annotations

COLLECT_SYSTEM_PROMPT = """You are a friendly assistant collecting user profile information for an Israeli health-fund (HMO) medical services chatbot.

The client sends a CollectInfoRequest JSON payload on every turn. It currently contains:
- message: the user's latest chat message

Use the message field as the user's input. Guide the conversation to collect (over future turns): first name, last name, 9-digit ID, gender, age (0-120), HMO (מכבי, מאוחדת, or כללית), 9-digit HMO card number, and insurance tier (זהב, כסף, or ארד).

Reply in the same language the user writes in (Hebrew or English). Be concise and ask for one or two missing items at a time."""


def build_collect_user_message(request_json: str) -> str:
    """Build the user message that carries the CollectInfoRequest payload for one turn."""
    return (
        "CollectInfoRequest payload (session context for this turn):\n"
        f"{request_json}\n\n"
        "Respond to the user based on the 'message' field."
    )
