"""
Given a docx file, extract its content to a structured JSON file to be used for chunking
and generating embeddings.
"""

from dataclasses import dataclass
import psycopg
from sentence_transformers import SentenceTransformer
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import json

# Helper to preserve order of block-level elements
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

USE_PLACEHOLDER_IMAGES = True


@dataclass
class Heading:
    level: int
    text: str


class DocumentExtract:
    doc: Document
    current_section: list[Heading]
    elements: list

    def __init__(self, file_path: str) -> None:
        self.doc = Document(file_path)
        self.current_section = []
        self.elements = []

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

    def _extract_table(self, table: Table):
        """
        Process a table block item.
        """

        # There are a couple special case tables that we want to detect

        # Chapter titles are in headers
        # Goals
        """
        {
            type: "2050_goals_table",
            section: ""
            values: {
                livable: "",
                resilient: "",
                equitable: ""
            }
        }
        """

        # 3 Things to know
        """
        {
            type: "3_facts_table",
            section: "",
            facts: [
                {
                    title: ""
                    text: ""
                }
            ]
        }
        """
        # 3 Things public engagement told us
        """
        same as above but type == 3_public_engagement
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

                for para in cell.paragraphs:
                    for run in para.runs:
                        font = run.font
                        style_info = {
                            "text": run.text,
                            "font_name": font.name,
                            "font_size": font.size.pt if font.size else None,
                            "bold": font.bold,
                            "italic": font.italic,
                        }

                text = cell.text.strip()
                if text:
                    cols.append(text)
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
