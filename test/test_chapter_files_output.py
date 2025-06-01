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
