"""
Given a docx file, extract its content to a structured JSON file to be used for chunking
and generating embeddings.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from zipfile import ZipFile

import psycopg
from docx import Document

# Helper to preserve order of block-level elements
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree
from sentence_transformers import SentenceTransformer

USE_PLACEHOLDER_IMAGES = True


@dataclass
class Heading:
    level: int
    text: str


# Step 1: Load the numbering.xml file from the .docx archive
def load_numbering_xml(docx_path):
    with ZipFile(docx_path) as docx_zip:
        numbering_xml = docx_zip.read("word/numbering.xml")
        return etree.fromstring(numbering_xml)


# Step 2: Parse numbering definitions from numbering.xml
def parse_numbering_definitions(numbering_tree):
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    numId_to_abstractId = {}
    abstractId_to_levels = {}

    for num in numbering_tree.findall(".//w:num", namespaces=ns):
        numId = num.get(f"{{{ns['w']}}}numId")
        abstractNumId_el = num.find(".//w:abstractNumId", namespaces=ns)
        if abstractNumId_el is not None:
            numId_to_abstractId[numId] = abstractNumId_el.get(f"{{{ns['w']}}}val")

    for abstractNum in numbering_tree.findall(".//w:abstractNum", namespaces=ns):
        abstractId = abstractNum.get(f"{{{ns['w']}}}abstractNumId")
        levels = {}
        for lvl in abstractNum.findall(".//w:lvl", namespaces=ns):
            ilvl = lvl.get(f"{{{ns['w']}}}ilvl")
            numFmt = lvl.find("w:numFmt", namespaces=ns)
            lvlText = lvl.find("w:lvlText", namespaces=ns)
            levels[ilvl] = {
                "numFmt": numFmt.get(f"{{{ns['w']}}}val")
                if numFmt is not None
                else None,
                "lvlText": lvlText.get(f"{{{ns['w']}}}val")
                if lvlText is not None
                else None,
            }
        abstractId_to_levels[abstractId] = levels

    return numId_to_abstractId, abstractId_to_levels


class DocumentExtract:
    doc: Document
    current_section: list[Heading]
    numbering_tree: Any
    elements: list

    def __init__(self, file_path: str) -> None:
        self.doc = Document(file_path)
        self.current_section = []
        self.elements = []
        self.numbering_tree = load_numbering_xml(file_path)
        self.numId_to_abstractId, self.abstractId_to_levels = (
            parse_numbering_definitions(self.numbering_tree)
        )

        print("abstractid to levels")
        print(self.abstractId_to_levels)

        print("numlevels to abstrract id")
        print(self.numId_to_abstractId)
        self.list_counters = defaultdict(lambda: defaultdict(int))

    def extract(self):
        """
        Process the document, extracting the elements into JSON.
        """
        elements = []

        for block in self._iter_block_items():
            if isinstance(block, Paragraph):
                data = self._extract_paragraph(block)
                if data and data["text"]:  # Skip empty paragraphs
                    elements.append(data)
            elif isinstance(block, Table):
                data = self._extract_table(block)
                if data:
                    elements.append(data)
            else:
                raise Exception(f"Unexpected instance item {block}")

        self.elements = elements

    def encode(self):
        if not self.elements:
            return

        model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good for retrieval

        def prepare(chunk):
            if chunk.get("section_path", None):
                prefix = "\n".join(chunk.get("section_path"))
                return f"{prefix}\n{chunk['text']}"
            return chunk["text"]

        texts = [prepare(chunk) for chunk in self.elements]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.tolist()

        assert len(embeddings) == len(self.elements), (
            "Expected embeddings and elements to be the same length"
        )

        for i, e in enumerate(self.elements):
            e["embedding"] = embeddings[i]

    def dump_to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.elements, f, indent=2)

    def dump_to_db(self):
        conn = psycopg.connect(
            dbname="db", user="admin", password="password", host="localhost", port=5431
        )
        cur = conn.cursor()

        for e in self.elements:
            cur.execute(
                """
                INSERT INTO plan_chunks (content_type, content, section_path, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    e["type"],
                    e["text"],
                    json.dumps(e.get("section_path", None)),
                    e["embedding"],
                ),
            )

        conn.commit()
        cur.close()
        conn.close()

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

    # Track list state and render list numbers
    def build_list_number(self, numId, ilvl):
        abstract_id = self.numId_to_abstractId[numId]
        levels = self.abstractId_to_levels[abstract_id]

        # Pull the lvlText and format from the current level
        lvlText = levels[str(ilvl)]["lvlText"]
        label = lvlText

        ilvl = int(ilvl)
        # Reset deeper levels
        for lvl in range(ilvl + 1, 9):
            self.list_counters[numId][lvl] = 0
        # Increment current level
        self.list_counters[numId][ilvl] += 1

        print(self.list_counters)

        # Replace each %n with the formatted value from list_counters
        for n in range(1, 10):  # %1 to %9
            if f"%{n}" in label:
                lvl_index = n - 1
                counter = self.list_counters[numId].get(lvl_index, 0)
                # TODO - there are different format types.
                # fmt_type = levels.get(str(lvl_index), {}).get("numFmt", "decimal")
                label = label.replace(f"%{n}", str(counter))

        return label

    def _extract_table(self, table: Table):
        """
        Process a table block item.

        This includes some handling for special cases.
        """

        seen_cells = set()

        rows = []

        styles = []

        for row in table.rows:
            cols = []
            for cell in row.cells:
                # Avoid duplicate references due to merged cells
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)

                list_num = None
                style_info = None
                for para in cell.paragraphs:
                    ### Try to get list ids
                    p = para._p
                    numPr = p.find(".//w:numPr", para._element.nsmap)
                    if numPr is not None:
                        ilvl_el = numPr.find("w:ilvl", para._element.nsmap)
                        numId_el = numPr.find("w:numId", para._element.nsmap)
                        if ilvl_el is not None and numId_el is not None:
                            ilvl = ilvl_el.get(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                            )
                            numId = numId_el.get(
                                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val"
                            )
                            list_num = self.build_list_number(numId, ilvl)

                    ### Get styles - just the first run is fine
                    for run in para.runs:
                        font = run.font
                        style_info = {
                            "text": run.text,
                            "font_name": font.name,
                            "font_size": font.size.pt if font.size else None,
                            "bold": font.bold,
                            "italic": font.italic,
                        }

                text = cell.text.strip() or list_num
                if text:
                    cols.append(text)
                    if style_info:
                        styles.append(style_info)
            if cols:
                rows.append(cols)

        if not rows:
            return None

        # Check if this is actually a big chapter header
        if all(
            [
                s["font_name"] == "Bumper Sticker" and float(s["font_size"]) >= 26.0
                for s in styles
            ]
        ) and (len(rows) == 1 and len(rows[0]) == 2):
            self.current_section.append(Heading(1, rows[0][1]))
            return {"type": "heading", "level": 1, "text": rows[0][1]}

        # Is this actually the 2050 goals table?
        if rows and rows[0] and rows[0][0].startswith("Goals: In 2050"):
            assert len(rows) == 7, (
                f"Expected 7, found {len(rows[0])} - num rows {len(rows)}"
            )
            assert rows[1][0] == "Livable", f"Expected Livable, found {rows[1][0]}"
            assert rows[3][0] == "Resilient", f"Expected Resilient, found {rows[3][0]}"
            assert rows[5][0] == "Equitable", f"Expected Equitable, found {rows[5][0]}"

            return {
                "type": "2050_goals",
                "text": rows[0][0],
                "section": self.current_section[0].text,
                "values": {
                    "livable": rows[2][0].strip(),
                    "resilient": rows[4][0].strip(),
                    "equitable": rows[6][0].strip(),
                },
            }

        # Is this our 3 Facts table?
        if rows and rows[0] and rows[0][0].startswith("Three Things"):
            assert len(rows) == 7, (
                f"For 3 Facts Table - Expected 7, found {len(rows[0])} - num rows {len(rows)}"
            )

            return {
                "type": "3_public_engagement_findings"
                if "Public Engagement" in rows[0][0]
                else "3_facts",
                "text": rows[0][0],
                "section": self.current_section[0].text,
                "facts": [
                    {"title": rows[1][0], "text": rows[2][0]},
                    {"title": rows[3][0], "text": rows[4][0]},
                    {"title": rows[5][0], "text": rows[6][0]},
                ],
            }

        # Is this the objectives tables

        return {
            "type": "table",
            "rows": rows,
        }

    def _extract_paragraph(self, paragraph: Paragraph):
        """
        Parse headers and text body, keeping the current section path up to date.
        """

        if not paragraph.text.strip():
            return

        if paragraph.style.name == "Section Heading":
            level = 2

            # Adjust our current section.
            while self.current_section and self.current_section[-1].level >= level:
                self.current_section.pop()

            curr_heading = Heading(level, paragraph.text.strip())
            self.current_section.append(curr_heading)
            return {
                "type": "heading",
                "text": curr_heading.text,
                "level": curr_heading.level,
                "section": self.current_section[0].text,
            }
        elif paragraph.style.name in ["Normal", "No Spacing"]:
            return {
                "type": "paragraph",
                "paragraph_style": paragraph.style.name,
                "text": paragraph.text.strip(),
                "section_path": [s.text for s in self.current_section],
            }
        else:
            raise Exception(
                f"Unknown paragraph style {paragraph.style.name}. Content: {paragraph.text.strip()}"
            )


# Example usage: `python extract.py files/docx_test.docx`
if __name__ == "__main__":
    import argparse
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

    ex.extract()
    # ex.encode()
    ex.dump_to_file(output_file)
    # ex.dump_to_db()

    print(f"Content extracted and saved to: {output_file}")
