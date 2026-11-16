import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.6-flash"

MEETING_NOTES = """
The project team met on Monday.
Maya will send the revised project timeline to the team by Friday.
Omar needs to book a room for the user-testing session; no due date was given.
Priya will share the accessibility checklist by November 20.
The team also discussed improving onboarding, but did not assign an owner or deadline.
"""

STAGES = [
    (
        "1. Vague prompt",
        "Review these meeting notes and list the action items.\n\n"
        f"Meeting notes:\n{MEETING_NOTES}",
        False,
    ),
    (
        "2. Role and instructions",
        "You are a careful meeting-notes assistant. Extract only explicit action "
        "items from the notes. For each item, identify the owner, task, and "
        "deadline. Use \"Not specified\" when the notes do not provide one; "
        "do not invent details.\n\n"
        f"Meeting notes:\n{MEETING_NOTES}",
        False,
    ),
    (
        "3. JSON output",
        "You are a careful meeting-notes assistant. Extract only explicit action "
        "items. Do not turn discussion topics without an assigned action into "
        "tasks. Do not invent details; use \"Not specified\" for a missing owner "
        "or deadline. Return only JSON in this exact shape:\n"
        '{"action_items":[{"owner":"...","task":"...","deadline":"..."}]}\n\n'
        f"Meeting notes:\n{MEETING_NOTES}",
        True,
    ),
    (
        "4. Step-by-step extraction",
        "You are a careful meeting-notes assistant. Extract only explicit action "
        "items. Do not turn discussion topics without an assigned action into "
        "tasks. Do not invent details; use \"Not specified\" for a missing owner "
        "or deadline. First provide a brief, auditable summary of the extraction "
        "steps (not private chain-of-thought), then return only JSON in this shape:\n"
        '{"reasoning_steps":["..."],"action_items":'
        '[{"owner":"...","task":"...","deadline":"..."}]}\n\n'
        f"Meeting notes:\n{MEETING_NOTES}",
        True,
    ),
]


def generate(prompt: str, json_output: bool) -> str:
    config: dict[str, Any] = {}
    if json_output:
        config["response_mime_type"] = "application/json"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(**config),
    )
    return (response.text or "").strip()


def summarize_output(text: str) -> tuple[bool, int, bool]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, 0, False

    if not isinstance(data, dict):
        return True, 0, False

    items = data.get("action_items")
    valid_items = isinstance(items, list) and all(
        isinstance(item, dict)
        and all(key in item for key in ("owner", "task", "deadline"))
        for item in items
    )
    return True, len(items) if isinstance(items, list) else 0, valid_items


def main() -> None:
    results: list[tuple[str, str]] = []
    for title, prompt, json_output in STAGES:
        result = generate(prompt, json_output)
        results.append((title, result))
        print(f"\n--- {title} ---\n{result}")

    print("\n--- Comparison ---")
    print(f"{'Stage':<30} {'Valid JSON':<12} {'Action items':<14} Required fields")
    for title, result in results:
        is_json, item_count, has_fields = summarize_output(result)
        fields = "owner, task, deadline" if has_fields else "not reliably structured"
        print(f"{title:<30} {str(is_json):<12} {item_count:<14} {fields}")

    print(
        "\nThe prompts become progressively more specific: role/instructions "
        "reduce ambiguity, JSON makes the result machine-readable, and the final "
        "stage adds a concise explanation of extraction decisions."
    )


if __name__ == "__main__":
    main()
