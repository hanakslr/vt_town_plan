"""
Parser for action tables in the document.
"""

import re
from typing import Dict, List

from models import Action, Objective, Strategy


def parse_action_table(rows: List[List[str]]) -> Dict[str, List]:
    """
    Parse an action table into a structured format.

    Action tables follow this format:
    - Objectives (2.A, 2.B, etc.)
    - Strategies (2.1, 2.2, etc.)
    - Actions (2.1.1, 2.1.2, etc.) with responsibility, time frame, cost, etc.

    Returns a dictionary with:
    {
        "objectives": [Objective objects],
        "strategies": [Strategy objects with nested Action objects]
    }
    """
    result = {
        "objectives": [],
        "strategies": [],
    }

    # Regular expressions for identifying different components
    objective_label_pattern = re.compile(r"^(\d+\.[A-Z])$")
    strategy_label_pattern = re.compile(r"^(\d+\.\d+)$")
    action_label_pattern = re.compile(r"^((\d+\.\d+)\.\d+)$")

    for row in rows:
        if not row:
            continue

        # Title rows
        if row[0].lower().replace(",", "") in [
            "objectives strategies and Actions",
            "objectives",
            "strategies",
        ]:
            continue

        cell_text = row[0] if len(row) > 0 else ""

        objective_label_match = objective_label_pattern.match(row[0])

        # Check if this is an objective
        if objective_label_match:
            if len(row) == 2:
                result["objectives"].append(Objective(label=row[0], text=row[1]))
                continue
            else:
                raise Exception("Unexpected row in Objectives", row)

        # Check if this is a strategy
        strategy_match = strategy_label_pattern.match(cell_text)
        if strategy_match:
            # Check to see if there is already a strategy
            existing_strat = [s for s in result["strategies"] if s.label == row[0]]

            if existing_strat:
                # If we have an existing strategy, we have a label and maybe actions,
                # but maybe we don't have text
                existing_strat = existing_strat[0]
                if not existing_strat.text and len(row) > 1:
                    existing_strat.text = row[1]
            else:
                result["strategies"].append(
                    Strategy(label=row[0], text=row[1], actions=[])
                )
            continue

        # Check if this is an action
        action_match = action_label_pattern.match(cell_text)
        if action_match:
            assert len(row) >= 2, f"expected row to have label and text {row}"
            label = row[0]
            text = row[1]

            # Extract additional fields if available
            responsibility = row[2] if len(row) > 2 else ""
            time_frame = row[3] if len(row) > 3 else ""
            cost = row[4] if len(row) > 4 else ""

            # Check for special markers
            starred = "★" in cell_text or "*" in cell_text
            multiple_strategies = "†" in cell_text or "†" in cell_text

            # Create Action object with None for optional fields by default
            action_kwargs = {
                "label": label,
                "text": text,
                "responsibility": responsibility,
                "time_frame": time_frame,
                "cost": cost,
            }

            # Only set these fields to True if they're present, otherwise leave as None
            # They'll be omitted from the JSON output when they're None or False
            if starred:
                action_kwargs["starred"] = True

            if multiple_strategies:
                action_kwargs["multiple_strategies"] = True

            action = Action(**action_kwargs)
            strategy_label_from_action = action_match.group(2)

            # Find the related strategy. If it doesn't exist make a placeholder for it.
            s = [
                s for s in result["strategies"] if strategy_label_from_action == s.label
            ]
            if s:
                s[0].actions.append(action)
            else:
                result["strategies"].append(
                    Strategy(
                        label=strategy_label_from_action, text="", actions=[action]
                    )
                )

            continue

    return result
