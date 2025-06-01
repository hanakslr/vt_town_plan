import json
from pathlib import Path

import pytest

from extract import DocumentExtract

INPUT_PATH = "files/chapters"
OUTPUT_PATH = "output/chapters"


def test_chapter_files_extraction_and_structure():
    """Test that:
    1. Extraction process runs successfully on all input files
    2. Output files have the correct structure:
       - First element is a header
       - Header has a chapter_number field
    """
    # Directory containing the input files
    input_dir = Path(INPUT_PATH)
    # Create a test-specific output directory that won't be cleaned up
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    # Get all input files (assuming they're .docx files)
    input_files = list(input_dir.glob("*.docx"))

    # Ensure we found some files to test
    assert len(input_files) > 0, "No input files found in input directory"

    # Process each input file
    for input_file in input_files:
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
