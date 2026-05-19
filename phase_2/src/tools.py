from __future__ import annotations

from typing import Any

SEARCH_HMO_KNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_hmo_knowledge",
        "description": (
            "Search the Israeli health-fund knowledge base for medical service coverage, "
            "benefits, and contact details. Use when the user asks about HMO services, "
            "insurance tiers, discounts, or phone numbers — not for greetings or profile edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query reflecting what to look up, in the user's language",
                },
            },
            "required": ["query"],
        },
    },
}
