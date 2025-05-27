"""
Given a docx file, extract its content to a structured JSON file to be used for chunking
and generating embeddings.
"""

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.shape import InlineShape
import base64

# Helper to preserve order of block-level elements
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

USE_PLACEHOLDER_IMAGES = True

class DocumentExtract:
    doc: Document

    def __init__(self, file_path: str) -> None:
        self.doc = Document(file_path)

    def extract(self):
        """
        Process the document, extracting the elements into JSON.
        """
        elements = []

        for block in self._iter_block_items():
            if isinstance(block, Paragraph):
                data = self._extract_paragraph(block)
                if data["text"]:  # Skip empty paragraphs
                    elements.append(data)
            elif isinstance(block, Table):
                elements.append(self._extract_table(block))
            else:
                raise Exception(f"Unexpected instance item {block}")

        return elements
    
    def _iter_block_items(self):
        """Yield paragraphs and tables in document order"""
        parent_elm = self.doc._element.body
        for child in parent_elm.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, self.doc)
            elif child.tag == qn("w:tbl"):
                yield Table(child, self.doc)
            elif child.tag == qn("w:sectPr"):
                # This is formatting width/height/margin info we don't care about right now
                pass
            else:
                raise Exception(f"Unhandled tag -  {child.tag}")


    def _extract_table(self, table: Table):
        """
        Process a table block item.
        """
        return {
            "type": "table",
            "rows": [[cell.text.strip() for cell in row.cells] for row in table.rows],
        }


    def _extract_paragraph(self, paragraph: Paragraph):
        """
        Add the section path.
        """
        style_name = paragraph.style.name.lower()
        if style_name.startswith("heading"):
            try:
                level = int(style_name.replace("heading", "").strip())
            except ValueError:
                level = 1
            return {"type": "heading", "level": level, "text": paragraph.text.strip()}
        else:
            return {"type": "paragraph", "text": paragraph.text.strip(), "section_path": []} # Implement section_path!
        





# Example usage: `python extract.py files/docx_test.docx`
if __name__ == "__main__":
    import json
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Extract content from a Word document."
    )
    parser.add_argument(
        "input_file", help="Path to the input Word document (.docx file)"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate output filename based on input filename
    input_filename = Path(args.input_file).stem
    output_file = output_dir / f"{input_filename}.json"

    ex = DocumentExtract(args.input_file)

    content = ex.extract()

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

    print(f"Content extracted and saved to: {output_file}")
