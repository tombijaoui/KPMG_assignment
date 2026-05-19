from __future__ import annotations

from typing import Any

SEARCH_HMO_KNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_hmo_knowledge",
        "description": (
            "Look up Israeli health-fund information on service coverage, benefits, and "
            "contact details. Helpful when the member asks about HMO services, insurance "
            "tiers, discounts, or phone numbers. Usually skip for greetings only."
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

QA_TOOLS: list[dict[str, Any]] = [SEARCH_HMO_KNOWLEDGE_TOOL]
