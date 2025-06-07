"""
Parser for action tables in the document.
"""

import re
from typing import Dict, List


def parse_action_table(rows: List[List[str]]) -> Dict:
    """
    Parse an action table into a structured format.

    Action tables follow this format:
    - Objectives (2.A, 2.B, etc.)
    - Strategies (2.1, 2.2, etc.)
    - Actions (2.1.1, 2.1.2, etc.) with responsibility, time frame, cost, etc.

    Returns a dictionary with:
    {
        "objectives": [{label, text}, ...],
        "strategies": [{
            label,
            text,
            actions: [{
                label,
                text,
                responsibility,
                time_frame,
                cost,
                starred,
                multiple_strategies
            }, ...]
        }, ...]
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
                result["objectives"].append({"label": row[0], "text": row[1]})
                continue
            else:
                raise Exception("Unexpected row in Objectives", row)

        # Check if this is a strategy
        strategy_match = strategy_label_pattern.match(cell_text)
        if strategy_match:
            result["strategies"].append(
                {"label": row[0], "text": row[1], "actions": []}
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

            action = {
                "label": label,
                "text": text,
                "responsibility": responsibility,
                "time_frame": time_frame,
                "cost": cost,
            }

            if starred:
                action["starred"] = True

            if multiple_strategies:
                action["multiple_strategies"] = True

            strategy_label_from_action = action_match.group(2)

            # Find the related strategy. If it doesn't exist make a placeholder for it.
            s = [
                s
                for s in result["strategies"]
                if strategy_label_from_action == s["label"]
            ]
            if s:
                s[0]["actions"].append(action)
            else:
                result["strategies"].append(
                    {"label": strategy_label_from_action, "actions": [action]}
                )

            continue

    return result
