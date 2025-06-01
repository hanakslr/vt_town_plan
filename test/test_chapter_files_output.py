import json
from pathlib import Path
import re

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
