"""OpenAI-format tool schemas.

We hand-write these instead of using a framework so the prompt-engineering
contract is visible. Free-tier models can be picky about JSON schemas;
keeping them tight and explicit improves tool-call reliability.
"""
from __future__ import annotations

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jd_requirements",
            "description": (
                "Parse the job description into ranked requirements. Call this "
                "exactly once at the start, before drafting any bullets."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search the user's career history for facts relevant to a query. "
                "Use this to find concrete past experiences that match a JD requirement. "
                "Returns a list of facts with stable IDs you can cite later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The information need, e.g. 'Python production systems'.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of hits to return (1-10).",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_claim",
            "description": (
                "Verify that a drafted bullet is grounded in the supplied fact IDs. "
                "Returns grounded=true only if every factual claim in the bullet "
                "(numbers, employer names, technologies, outcomes) is supported by "
                "the cited facts. CALL THIS BEFORE emit_bullet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bullet": {"type": "string"},
                    "supporting_fact_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["bullet", "supporting_fact_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_bullet",
            "description": (
                "Submit a finalised tailored bullet. Only call after verify_claim "
                "returned grounded=true. Provide every fact_id you used."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "description": "Which resume section this bullet belongs to (e.g. 'summary', 'experience.acme').",
                    },
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "fact_ids from search_history results that support this bullet.",
                    },
                },
                "required": ["section", "text", "citations"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that tailoring is complete. Call this after every requested "
                "section has either an emitted bullet or a reason it could not be grounded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                },
                "required": ["summary"],
            },
        },
    },
]
