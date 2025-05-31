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
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml import etree
from sentence_transformers import SentenceTransformer

from parsers.table_parsers import TableMerger, TableParserFactory, TableStyles

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
        self.list_counters = defaultdict(lambda: defaultdict(int))

    def extract(self):
        """
        Process the document, extracting the elements into JSON.
        """
        elements = []
        previous_table = None

        for block in self._iter_block_items():
            if isinstance(block, Paragraph):
                data = self._extract_paragraph(block)
                if data and data["text"]:  # Skip empty paragraphs
                    elements.append(data)
                    # We keep previous_table even after paragraphs now, to allow merging
                    # action tables that might be separated by explanatory paragraphs

            elif isinstance(block, Table):
                data = self._extract_table(block)
                if data:
                    # Check if we should merge with a previous table
                    # This handles both regular tables and action tables now
                    if (previous_table and 
                        TableMerger.should_merge(previous_table, data)):
                        
                        # Merge with previous table instead of adding a new one
                        previous_table = TableMerger.merge_tables(previous_table, data)
                        
                        # Replace the last element with the merged table if it's there
                        # We need to find the last occurrence of the previous table type
                        for i in range(len(elements) - 1, -1, -1):
                            if elements[i].get("type") == previous_table.get("type"):
                                elements[i] = previous_table
                                break
                    else:
                        # Add as a new element
                        elements.append(data)
                        previous_table = data
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
        Process a table block item with the new table parser system.

        Handles special cases and table merging.
        """
        # Step 1: Do a preliminary extraction of rows and styles to determine the table type
        rows = []
        styles = []
        seen_cells = set()

        # # Debug tables with potential objectives content
        # first_cell_text = ""
        # if table.rows and table.rows[0].cells:
        #     first_cell_text = table.rows[0].cells[0].text.strip()

        # # Check if this looks like an objectives table or contains objectives
        # if first_cell_text == "Objectives" or any(
        #     cell.text.strip().startswith("5.")
        #     for row in table.rows
        #     for cell in row.cells
        # ):
        #     # Print detailed cell information
        #     print("\n--- CELL DETAILS ---")
        #     for i, row in enumerate(table.rows):
        #         print(f"Row {i}:")
        #         for j, cell in enumerate(row.cells):
        #             print(f"  Cell {j}: Text='{cell.text.strip()}'")

        #             # Check for merged cells
        #             vmerge = cell._tc.find(".//w:vMerge", cell._tc.nsmap)
        #             is_vertical_merge = vmerge is not None
        #             vmerge_val = vmerge.get(qn("w:val")) if vmerge is not None else None
        #             is_origin = vmerge_val == "restart" or vmerge is None

        #             gridspan = cell._tc.find(".//w:gridSpan", cell._tc.nsmap)
        #             gridspan_val = (
        #                 gridspan.get(qn("w:val")) if gridspan is not None else None
        #             )

        #             print(f"    Vertically merged: {is_vertical_merge}")
        #             print(f"    Merge origin: {is_origin}")
        #             print(f"    vMerge value: {vmerge_val}")
        #             print(f"    Grid span: {gridspan_val}")

        #             # Look for content in paragraphs
        #             print(f"    Paragraphs: {len(cell.paragraphs)}")
        #             for p_idx, para in enumerate(cell.paragraphs):
        #                 print(f"      Paragraph {p_idx}: '{para.text}'")
        #     print("===================================\n\n")

        # Extract basic rows and styles to identify table type
        for row in table.rows:
            cols = []
            for cell in row.cells:
                cell_id = id(cell._tc)
                if cell_id in seen_cells:
                    continue
                seen_cells.add(cell_id)

                text = cell.text.strip()

                # Get style info from the first run in the first paragraph
                style_info = None
                for para in cell.paragraphs:
                    for run in para.runs:
                        font = run.font
                        style_info = TableStyles(
                            text=run.text,
                            font_name=font.name,
                            font_size=font.size.pt if font.size else None,
                            bold=font.bold,
                            italic=font.italic,
                        )
                        break
                    if style_info:
                        break

                if text:
                    cols.append(text)
                    if style_info:
                        styles.append(style_info)
            if cols:
                rows.append(cols)

        if not rows:
            return None

        # Step 2: Create the appropriate parser based on the extracted data
        section_texts = [s.text for s in self.current_section]
        parser = TableParserFactory.create_parser(
            table=table,
            rows=rows,
            styles=styles,
            current_section=section_texts,
            list_number_generator=self.build_list_number,
        )

        # Step 3: Parse the table
        result = parser.parse()

        # Step 4: Handle special cases
        if result and result.get("type") == "heading" and result.get("level") == 1:
            # If this is a chapter header, update the current section
            self.current_section.append(Heading(1, result["text"]))

        # Step 5: Check if this table should be merged with the previous one
        if (
            result
            and result.get("type") == "table"
            and self.elements
            and self.elements[-1].get("type") == "table"
        ):
            # Check if tables should be merged (same column count, consecutive in document, etc.)
            if TableMerger.should_merge(self.elements[-1], result):
                # Merge with previous table instead of adding a new one
                self.elements[-1] = TableMerger.merge_tables(self.elements[-1], result)
                return None  # Signal that we've merged with a previous table

        return result

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
        elif paragraph.style.name in ["Normal", "No Spacing", "paragraph"]:
            return {
                "type": "paragraph",
                "paragraph_style": paragraph.style.name,
                "text": paragraph.text.strip(),
                "section_path": [s.text for s in self.current_section],
            }
        elif paragraph.style.name == "Caption":
            return {
                "type": "caption",
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
