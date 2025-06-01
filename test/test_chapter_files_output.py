import json
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
def input_file(request):
    """Fixture that provides each input file as a separate test case."""
    return request.param


def test_chapter_file_extraction_and_structure(input_file):
    """Test that a single chapter file:
    1. Extracts successfully
    2. Output file has the correct structure:
       - First element is a header
       - Header has a chapter_number field
    """
    # Create output directory
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    # Create output filename
    output_file = output_dir / f"{input_file.stem}.json"

    try:
        # Run extraction
        ex = DocumentExtract(str(input_file))
        ex.extract()
        ex.dump_to_file(output_file)

        # Validate output file
        with open(output_file, "r") as f:
            data = json.load(f)

            # Check that file is not empty
            assert len(data) > 0, f"Extracted file {output_file} is empty"

            # Check first element is a header
            first_element = data[0]
            assert first_element["type"] == "heading", (
                f"First element in {output_file} is not a header"
            )

            # Check header has chapter_number
            assert "chapter_number" in first_element, (
                f"Header in {output_file} missing chapter_number field"
            )

            # Check chapter_number is a string
            assert isinstance(first_element["chapter_number"], str), (
                f"chapter_number in {output_file} is not a string"
            )

    except Exception as e:
        pytest.fail(f"Error processing {input_file}: {str(e)}")
