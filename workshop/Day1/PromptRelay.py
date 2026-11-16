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
        "items. First identify commitment statements such as \"will send\" and "
        "\"needs to\", then extract the owner, task, and deadline from each one. "
        "Do not turn discussion topics without an assigned action into "
        "tasks. Do not invent details; use \"Not specified\" for a missing owner "
        "or deadline. Include a brief, auditable summary of those steps, not "
        "private chain-of-thought. Return JSON in this shape:\n"
        '{"reasoning_steps":["..."],"action_items":'
        '[{"owner":"...","task":"...","deadline":"..."}]}\n\n'
        f"Meeting notes:\n{MEETING_NOTES}",
        True,
    ),
]

EXPECTED_ACTIONS = [
    {
        "owner": "Maya",
        "task": "Send the revised project timeline",
        "task_terms": (("revised project timeline", "revised timeline"),
                       ("send", "deliver", "share", "provide")),
        "deadline_terms": (("friday",),),
    },
    {
        "owner": "Omar",
        "task": "Book a room for the user-testing session",
        "task_terms": (("book", "reserve", "schedule"), ("room",),
                       ("user-testing", "user testing", "testing session")),
        "deadline_terms": ((
            "not specified", "no due date", "no deadline", "not given",
            "none given", "no specific deadline",
        ),),
    },
    {
        "owner": "Priya",
        "task": "Share the accessibility checklist",
        "task_terms": (("accessibility checklist",),
                       ("share", "send", "provide", "circulate", "distribute")),
        "deadline_terms": (("november 20", "november 20th", "nov 20", "nov 20th", "11/20"),),
    },
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


def _contains_any(text: str, alternatives: tuple[str, ...]) -> bool:
    return any(term in text for term in alternatives)


def _contains_all_groups(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    return all(_contains_any(text, group) for group in groups)


def _output_records(text: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], False

    if not isinstance(data, dict) or not isinstance(data.get("action_items"), list):
        return [], False

    items = data["action_items"]
    valid_items = all(
        isinstance(item, dict)
        and all(isinstance(item.get(key), str) for key in ("owner", "task", "deadline"))
        for item in items
    )
    return [item for item in items if isinstance(item, dict)], valid_items


def _evaluate_output(text: str) -> dict[str, Any]:
    records, valid_json = _output_records(text)
    if records:
        normalized_records = [
            " ".join(str(value) for value in record.values()).casefold()
            for record in records
        ]
    else:
        normalized_records = []
        lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
        owner_positions = [
            (line_index, item["owner"].casefold())
            for line_index, line in enumerate(lines)
            for item in EXPECTED_ACTIONS
            if item["owner"].casefold() in line
        ]
        for position, (line_index, _) in enumerate(owner_positions):
            next_line = (
                owner_positions[position + 1][0]
                if position + 1 < len(owner_positions)
                else len(lines)
            )
            normalized_records.append(" ".join(lines[line_index:next_line]))

    matched_owners: set[str] = set()
    field_matches = 0
    for item in EXPECTED_ACTIONS:
        owner = item["owner"].casefold()
        for record in normalized_records:
            if owner not in record:
                continue
            owner_match = True
            task_match = _contains_all_groups(record, item["task_terms"])
            deadline_match = _contains_all_groups(record, item["deadline_terms"])
            field_matches += owner_match + task_match + deadline_match
            if task_match and deadline_match:
                matched_owners.add(owner)
            break

    if records:
        trap_found = any(
            "onboarding" in " ".join(str(value) for value in record.values()).casefold()
            for record in records
        )
        owners = [
            str(record.get("owner", "")).strip().casefold()
            for record in records
        ]
        consistency = (
            valid_json
            and len(owners) == len(set(owners))
            and all(owner in {item["owner"].casefold() for item in EXPECTED_ACTIONS}
                    for owner in owners)
        )
    else:
        trap_found = any(
            "onboarding" in line
            and not any(
                phrase in line
                for phrase in ("no action", "not assigned", "no owner", "no deadline")
            )
            for line in text.casefold().splitlines()
        )
        consistency = None

    correct_count = len(matched_owners)
    return {
        "valid_json": valid_json,
        "correct_count": correct_count,
        "complete": correct_count == len(EXPECTED_ACTIONS),
        "trap_avoided": not trap_found,
        "consistent": consistency,
        "field_accuracy": round(field_matches / (len(EXPECTED_ACTIONS) * 3) * 100),
    }


def main() -> None:
    results: list[tuple[str, str]] = []
    for title, prompt, json_output in STAGES:
        result = generate(prompt, json_output)
        results.append((title, result))
        print(f"\n--- {title} ---\n{result}")

    print("\n--- Expected action items ---")
    for item in EXPECTED_ACTIONS:
        print(
            f"- {item['owner']}: {item['task']} "
            f"(deadline: {item['deadline_terms'][0][0]})"
        )

    print("\n--- Comparison ---")
    print(
        f"{'Stage':<30} {'Accuracy':<12} {'Complete':<11} "
        f"{'Trap avoided':<14} {'Consistent':<12} {'Valid JSON'}"
    )
    for title, result in results:
        evaluation = _evaluate_output(result)
        print(
            f"{title:<30} "
            f"{evaluation['field_accuracy']}% "
            f"({evaluation['correct_count']}/3){'':<3} "
            f"{str(evaluation['complete']):<11} "
            f"{str(evaluation['trap_avoided']):<14} "
            f"{str(evaluation['consistent']) if evaluation['consistent'] is not None else 'N/A':<12} "
            f"{evaluation['valid_json']}"
        )

    print(
        "\nAccuracy is the share of the 9 expected owner/task/deadline fields "
        "matched. Completeness requires all three action items. Trap avoided "
        "means the unassigned onboarding topic was not reported as an action. "
        "Consistency checks structured outputs for valid, unique, expected owners."
    )


if __name__ == "__main__":
    main()
