"""Convert a small, synthetic event format to ATIF using Harbor's models.

This is a teaching example, not a parser for any vendor's native session format.
It never invokes an agent, a model API, or a shell command from the input.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from harbor.models.trajectories import Agent, FinalMetrics, Step, Trajectory


def convert_events(events: list[dict[str, Any]]) -> Trajectory:
    """Group model fragments by turn_id and attach results by call_id.

    Unknown events, duplicate calls/results, and orphan results are rejected.
    Calls without a result are retained to represent interrupted execution.
    Usage on repeated assistant fragments is a snapshot: the last wins.
    """
    normalized: list[dict[str, Any]] = []
    turns: dict[str, dict[str, Any]] = {}
    calls: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()

    def get_turn(turn_id: str) -> dict[str, Any]:
        if turn_id not in turns:
            turn = {
                "source": "agent",
                "message": "",
                "llm_call_count": 1,
                "extra": {"native_turn_id": turn_id},
            }
            turns[turn_id] = turn
            normalized.append(turn)
        return turns[turn_id]

    for event in events:
        kind = event["type"]
        if kind == "user":
            normalized.append({"source": "user", "message": event["text"]})
        elif kind == "assistant":
            turn = get_turn(event["turn_id"])
            turn["message"] += event.get("text", "")
            if event.get("reasoning"):
                turn["reasoning_content"] = event["reasoning"]
            if "usage" in event:
                # These synthetic usage fields are per-call snapshots, not deltas.
                turn["metrics"] = event["usage"]
        elif kind == "tool_call":
            call_id = event["call_id"]
            if call_id in calls:
                raise ValueError(f"Duplicate tool call: {call_id}")
            turn = get_turn(event["turn_id"])
            turn.setdefault("tool_calls", []).append(
                {
                    "tool_call_id": call_id,
                    "function_name": event["name"],
                    "arguments": event["arguments"],
                }
            )
            calls[call_id] = turn
        elif kind == "tool_result":
            call_id = event["call_id"]
            if call_id not in calls:
                raise ValueError(f"Orphan tool result: {call_id}")
            if call_id in completed:
                raise ValueError(f"Duplicate tool result: {call_id}")
            owner = calls[call_id]
            owner.setdefault("observation", {"results": []})["results"].append(
                {"source_call_id": call_id, "content": event["content"]}
            )
            completed.add(call_id)
        else:
            raise ValueError(f"Unsupported synthetic event type: {kind}")

    steps = [
        Step(step_id=index, **data)
        for index, data in enumerate(normalized, start=1)
    ]

    def total(field: str) -> int | None:
        agent_steps = [step for step in steps if step.source == "agent"]
        if not agent_steps:
            return None
        values = []
        for step in agent_steps:
            if step.metrics is None or (value := getattr(step.metrics, field)) is None:
                return None
            values.append(value)
        return sum(values)

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id="synthetic-demo-run",
        agent=Agent(name="teaching-agent", version="1.0", model_name="demo/model"),
        steps=steps,
        final_metrics=FinalMetrics(
            total_steps=len(steps),
            total_prompt_tokens=total("prompt_tokens"),
            total_completion_tokens=total("completion_tokens"),
            total_cached_tokens=total("cached_tokens"),
        ),
        extra={"synthetic": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Synthetic JSONL input")
    parser.add_argument("output", type=Path, help="ATIF JSON output")
    args = parser.parse_args()
    events = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectory = convert_events(events)
    serialized = json.dumps(trajectory.to_json_dict(), ensure_ascii=False, indent=2)
    # Validate the serialized artifact too, rather than only the in-memory graph.
    Trajectory.model_validate_json(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {len(trajectory.steps)} steps to {args.output}")


if __name__ == "__main__":
    main()
