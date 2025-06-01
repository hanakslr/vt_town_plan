import json
import re
from pathlib import Path

import pytest

from extract import DocumentExtract

INPUT_PATH = "files/chapters"
OUTPUT_PATH = "output/chapters"


def get_input_files():
    """Get all input files from the input directory."""
    input_dir = Path(INPUT_PATH)
    return list(input_dir.glob("*.docx"))


@pytest.fixture(params=get_input_files(), ids=lambda x: x.stem)
def extracted_file_data(request):
    """Fixture that provides each input file as a separate test case."""

    input_file = request.param
    # Create output directory
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    # Create output filename
    output_file = output_dir / f"{input_file.stem}.json"

    # Run extraction
    ex = DocumentExtract(str(input_file))
    ex.extract()
    ex.dump_to_file(output_file)

    with open(output_file, "r") as f:
        data = json.load(f)

    # Check that file is not empty
    assert len(data) > 0, "Extracted file is empty"

    return data


def test_chapter_has_main_header(extracted_file_data):
    """
    Check that the chapter has a chapter heading
    """

    # Check first element is a header
    first_element = extracted_file_data[0]
    assert first_element["type"] == "heading", "First element is not a header"

    # Check header has chapter_number
    assert "chapter_number" in first_element, "Header missing chapter_number field"

    # Check chapter_number is a string
    assert isinstance(first_element["chapter_number"], str), (
        "chapter_number is not a string"
    )


def test_chapter_has_2050_goals(extracted_file_data):
    """
    Check that the chapter has a valid 2050 goals block
    """
    goals_block = [elem for elem in extracted_file_data if elem["type"] == "2050_goals"]

    assert len(goals_block) == 1, (
        f"Expected to find 1 goals block, found {len(goals_block)}"
    )

    goals_block = goals_block[0]

    assert goals_block["section"], "Expected to find section"

    assert list(goals_block["values"].keys()) == ["livable", "resilient", "equitable"]

    for text in goals_block["values"].values():
        assert re.match(r"^[a-zA-Z]", text), f"Expected {text} to start with a letter"


def test_has_3_facts(extracted_file_data):
    """
    Check the chapter has a properly formatted 3 facts block
    """
    facts_block = [elem for elem in extracted_file_data if elem["type"] == "3_facts"]

    assert len(facts_block) == 1, (
        f"Expected to find 1 facts block but found {len(facts_block)}"
    )

    facts_block = facts_block[0]
    assert facts_block["section"], "Expected to find section"
    assert facts_block["text"] == "Three Things to Know"

    assert len(facts_block["facts"]) == 3

    for f in facts_block["facts"]:
        assert list(f.keys()) == ["title", "text"]
        assert f["title"] and f["text"]


def test_has_public_engagement(extracted_file_data):
    """
    Check the chapter has a properly formatted public engagement block
    """
    engagement_block = [
        elem
        for elem in extracted_file_data
        if elem["type"] == "3_public_engagement_findings"
    ]

    assert len(engagement_block) == 1, (
        f"Expected to find 1 public engagement block but found {len(engagement_block)}"
    )

    engagement_block = engagement_block[0]
    assert engagement_block["section"], "Expected to find section"
    assert engagement_block["text"] == "Three Things Public Engagement Told Us"

    assert len(engagement_block["facts"]) == 3

    for f in engagement_block["facts"]:
        assert list(f.keys()) == ["title", "text"]
        assert f["title"] and f["text"]


def test_actions_table(extracted_file_data):
    """
    Check the format of the objectives/strategies/actions table.
    """
    action_block = [
        elem for elem in extracted_file_data if elem["type"] == "action_table"
    ]

    assert len(action_block) == 1

    action_block = action_block[0]

    assert action_block["section"]

    chapter_block = [
        elem
        for elem in extracted_file_data
        if elem["type"] == "heading" and elem["level"] == 1
    ][0]
    chapter_number = chapter_block["chapter_number"]

    ## Objectives
    # Each objective should have a label and text. The labels and text should go in order.
    assert "objectives" in action_block, "Action block missing objectives"
    objectives = action_block["objectives"]
    assert len(objectives) > 0, "No objectives found"

    # Check each objective has required fields and correct format
    for i, obj in enumerate(objectives):
        assert list(obj.keys()) == ["label", "text"], (
            f"Objective {i} has unexpected fields"
        )
        assert obj["label"], f"Objective {i} missing label"
        assert obj["text"], f"Objective {i} missing text"

        # Check label format matches chapter number (e.g. "9.A", "9.B", etc.)
        assert re.match(rf"^{chapter_number}\.[A-Z]$", obj["label"]), (
            f"Objective {i} label {obj['label']} doesn't match expected format {chapter_number}.X"
        )

        # Check text starts with a letter
        assert re.match(r"^[a-zA-Z]", obj["text"]), (
            f"Objective {i} text should start with a letter"
        )

    # Check objectives are in order
    for i in range(len(objectives) - 1):
        curr_label = objectives[i]["label"]
        next_label = objectives[i + 1]["label"]
        assert curr_label < next_label, (
            f"Objectives out of order: {curr_label} before {next_label}"
        )

    ## Strategies & Actions
    def parse_label(label):
        """Convert a label like '12.2.10' into a tuple of integers (12, 2, 10) for proper numeric comparison."""
        return tuple(int(x) for x in label.split("."))

    assert "strategies" in action_block, "Action block missing strategies"
    strategies = action_block["strategies"]
    assert len(strategies) > 0, "No strategies found"

    # Check each strategy has required fields and correct format
    for i, strategy in enumerate(strategies):
        assert list(strategy.keys()) == ["label", "text", "actions"], (
            f"Strategy {i} has unexpected fields"
        )
        assert strategy["label"], f"Strategy {i} missing label"
        assert strategy["text"], f"Strategy {i} missing text"
        assert isinstance(strategy["actions"], list), (
            f"Strategy {i} actions should be a list"
        )

        # Check label format matches chapter number (e.g. "9.1", "9.2", etc.)
        assert re.match(rf"^{chapter_number}\.\d+$", strategy["label"]), (
            f"Strategy {i} label {strategy['label']} doesn't match expected format {chapter_number}.X"
        )

        # Check text starts with a letter
        assert re.match(r"^[a-zA-Z]", strategy["text"]), (
            f"Strategy {i} text should start with a letter"
        )

        # Check actions for this strategy
        actions = strategy["actions"]
        assert len(actions) > 0, f"Strategy {strategy['label']} has no actions"

        for j, action in enumerate(actions):
            # Check required fields
            required_fields = ["label", "text", "responsibility", "time_frame", "cost"]
            assert all(field in action for field in required_fields), (
                f"Action {j} in strategy {strategy['label']} missing required fields"
            )
            assert all(action[field] for field in required_fields), (
                f"Action {j} in strategy {strategy['label']} has empty required fields"
            )

            # Check action label format (e.g. "9.1.1", "9.1.2", etc.)
            strategy_label = strategy["label"]
            assert re.match(rf"^{strategy_label}\.\d+$", action["label"]), (
                f"Action {j} label {action['label']} doesn't match strategy {strategy_label}"
            )

            # Check text starts with a letter
            assert re.match(r"^[a-zA-Z]", action["text"]), (
                f"Action {j} in strategy {strategy['label']} text should start with a letter"
            )

            # Check optional fields if present
            if "starred" in action:
                assert isinstance(action["starred"], bool), (
                    "starred field should be boolean"
                )
            if "multiple_strategies" in action:
                assert isinstance(action["multiple_strategies"], bool), (
                    "multiple_strategies field should be boolean"
                )

    # Check strategies are in order
    for i in range(len(strategies) - 1):
        curr_label = strategies[i]["label"]
        next_label = strategies[i + 1]["label"]
        assert parse_label(curr_label) < parse_label(next_label), (
            f"Strategies out of order: {curr_label} before {next_label}"
        )

        # Check actions are in order within each strategy
        actions = strategy["actions"]
        for j in range(len(actions) - 1):
            curr_label = actions[j]["label"]
            next_label = actions[j + 1]["label"]
            assert parse_label(curr_label) < parse_label(next_label), (
                f"Actions out of order in strategy {strategy['label']}: {curr_label} before {next_label}"
            )
