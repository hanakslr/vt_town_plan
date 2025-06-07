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
    Check that the chapter has a chapter number and title
    """
    # Check that chapter_number is present in the structured document
    assert "chapter_number" in extracted_file_data, (
        "Document missing chapter_number field"
    )

    # Check that title is present in the structured document
    assert "title" in extracted_file_data, "Document missing title field"

    # Check chapter_number is a string
    assert isinstance(extracted_file_data["chapter_number"], str), (
        "chapter_number is not a string"
    )


def test_chapter_has_2050_goals(extracted_file_data):
    """
    Check that the chapter has a valid 2050 goals block
    """
    # Check that 2050_goals is present in the structured document
    assert "2050_goals" in extracted_file_data, "Document missing 2050_goals field"

    goals_block = extracted_file_data["2050_goals"]

    # Check that the goals block has a section
    assert "section" in goals_block, "Expected to find section in 2050_goals"

    # Check that the goals block has the expected values
    assert "values" in goals_block, "Expected to find values in 2050_goals"
    assert list(goals_block["values"].keys()) == ["livable", "resilient", "equitable"]

    # Check that each value starts with a letter
    for text in goals_block["values"].values():
        assert re.match(r"^[a-zA-Z]", text), f"Expected {text} to start with a letter"


def test_has_3_facts(extracted_file_data):
    """
    Check the chapter has a properly formatted 3 facts block as a top-level key
    """
    # Check that three_facts is present in the structured document
    assert "three_facts" in extracted_file_data, "Document missing three_facts field"

    facts_block = extracted_file_data["three_facts"]

    # Check that the facts block has a section
    assert "section" in facts_block, "Expected to find section in three_facts"
    assert facts_block["text"] == "Three Things to Know"

    # Check that the facts block has 3 facts
    assert "facts" in facts_block, "Expected to find facts in three_facts"
    assert len(facts_block["facts"]) == 3

    # Check each fact
    for f in facts_block["facts"]:
        assert "title" in f and "text" in f, "Fact missing title or text"
        assert f["title"] and f["text"], "Fact has empty title or text"


def test_has_public_engagement(extracted_file_data):
    """
    Check the chapter has a properly formatted public engagement block as a top-level key
    """
    # Check that public_engagement is present in the structured document
    assert "public_engagement" in extracted_file_data, (
        "Document missing public_engagement field"
    )

    engagement_block = extracted_file_data["public_engagement"]

    # Check that the engagement block has a section
    assert "section" in engagement_block, (
        "Expected to find section in public_engagement"
    )
    assert engagement_block["text"] == "Three Things Public Engagement Told Us"

    # Check that the engagement block has 3 facts
    assert "facts" in engagement_block, "Expected to find facts in public_engagement"
    assert len(engagement_block["facts"]) == 3

    # Check each fact
    for f in engagement_block["facts"]:
        assert "title" in f and "text" in f, "Fact missing title or text"
        assert f["title"] and f["text"], "Fact has empty title or text"


def test_actions_table(extracted_file_data):
    """
    Check the format of the objectives/strategies/actions table.
    """
    # Check that actions is present in the structured document
    assert "actions" in extracted_file_data, "Document missing actions field"

    action_block = extracted_file_data["actions"]

    # Check that the action block has a section
    assert "section" in action_block, "Expected to find section in actions"

    # Get chapter number from the structured document
    chapter_number = extracted_file_data["chapter_number"]

    ## Objectives
    # Each objective should have a label and text. The labels and text should go in order.
    assert "objectives" in action_block, "Action block missing objectives"
    objectives = action_block["objectives"]
    assert len(objectives) > 0, "No objectives found"

    # Check each objective has required fields and correct format
    for i, obj in enumerate(objectives):
        assert "label" in obj and "text" in obj, (
            f"Objective {i} missing required fields"
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
        required_strategy_fields = ["label", "text", "actions"]
        assert all(field in strategy for field in required_strategy_fields), (
            f"Strategy {i} missing required fields"
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
                f"Action {j} in strategy {strategy['label']} missing required fields. Has {action}. Expected: {required_fields}"
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
                # In the JSON output, 'starred' should only be present when it's True
                assert action["starred"] is True, (
                    "starred field should be True when present"
                )
            if "multiple_strategies" in action:
                # In the JSON output, 'multiple_strategies' should only be present when it's True
                assert action["multiple_strategies"] is True, (
                    "multiple_strategies field should be True when present"
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
