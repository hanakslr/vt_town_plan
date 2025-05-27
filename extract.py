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


def extract_table(table: Table):
    return {
        "type": "table",
        "rows": [[cell.text.strip() for cell in row.cells] for row in table.rows],
    }


def extract_paragraph(paragraph: Paragraph):
    style_name = paragraph.style.name.lower()
    if style_name.startswith("heading"):
        try:
            level = int(style_name.replace("heading", "").strip())
        except ValueError:
            level = 1
        return {"type": "heading", "level": level, "text": paragraph.text.strip()}
    else:
        return {"type": "paragraph", "text": paragraph.text.strip()}


def extract_image(image: InlineShape):
    image_data = image._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = image.part.related_parts[image_data]
    image_bytes = part.blob
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image",
        "image_base64": base64_data,
        "content_type": part.content_type,
    }


def extract_document_content(docx_path):
    doc = Document(docx_path)
    elements = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            data = extract_paragraph(block)
            if data["text"]:  # Skip empty paragraphs
                elements.append(data)
        elif isinstance(block, Table):
            elements.append(extract_table(block))

    # Handle inline images separately (python-docx does not treat them as block items)
    for shape in doc.inline_shapes:
        elements.append(extract_image(shape))

    return elements


def iter_block_items(parent):
    """Yield paragraphs and tables in document order"""
    parent_elm = parent._element.body
    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


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

    content = extract_document_content(args.input_file)

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

    print(f"Content extracted and saved to: {output_file}")
