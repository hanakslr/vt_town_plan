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

    current_objective = None
    current_strategy = None

    # Regular expressions for identifying different components
    objective_label_pattern = re.compile(r"^(\d+\.[A-Z])")
    strategy_pattern = re.compile(r"^(\d+\.\d+)\s+(.+)$")
    action_pattern = re.compile(r"^(\d+\.\d+\.\d+)\s+(.+)$")

    curr_section = None

    for row in rows:
        if not row:
            continue

        if row[0] == "Objectives, Strategies, and Actions":
            continue
        if row[0] == "Objectives":
            curr_section = "objectives"
            continue
        if row[0] == "Strategies":
            curr_section = "strategies"
            continue

        cell_text = row[0] if len(row) > 0 else ""
        print(row)

        if curr_section == "objectives":
            objective_label_match = objective_label_pattern.match(row[0])

            # Check if this is an objective
            if objective_label_match and len(row) == 2:
                current_objective = {"label": row[0], "text": row[1]}
                result["objectives"].append(current_objective)
                continue
            else:
                raise Exception("Unexpected row in Objectives", row)

        # Check if this is a strategy
        strategy_match = strategy_pattern.match(cell_text)
        if strategy_match:
            label = strategy_match.group(1)
            text = strategy_match.group(2)
            current_strategy = {"label": label, "text": text, "actions": []}
            result["strategies"].append(current_strategy)
            continue

        # Check if this is an action
        action_match = action_pattern.match(cell_text)
        if action_match and current_strategy:
            label = action_match.group(1)
            text = action_match.group(2)

            # Extract additional fields if available
            responsibility = row[1] if len(row) > 1 else ""
            time_frame = row[2] if len(row) > 2 else ""
            cost = row[3] if len(row) > 3 else ""

            # Check for special markers
            starred = "★" in cell_text or "*" in cell_text
            multiple_strategies = "†" in cell_text or "†" in cell_text

            action = {
                "label": label,
                "text": text,
                "responsibility": responsibility,
                "time_frame": time_frame,
                "cost": cost,
                "starred": starred,
                "multiple_strategies": multiple_strategies,
            }

            current_strategy["actions"].append(action)
            continue
    raise "stop"
    return result
