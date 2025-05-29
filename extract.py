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
                if data["text"]:  # Skip empty paragraphs
                    elements.append(data)
            elif isinstance(block, Table):
                elements.append(self._extract_table(block))
            else:
                raise Exception(f"Unexpected instance item {block}")

        self.elements = elements

    def encode(self):
        if not self.elements:
            return
        
        model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good for retrieval

        def prepare(chunk):
            if chunk.get("section_path", None):
                prefix =  "\n".join(chunk.get("section_path"))
                return f"{prefix}\n{chunk['text']}"
            return chunk["text"]
        
        texts = [prepare(chunk) for chunk in self.elements]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.tolist()

        assert len(embeddings) == len(self.elements), "Expected embeddings and elements to be the same length"

        for i, e in enumerate(self.elements):
            e["embedding"] = embeddings[i]

    def dump_to_file(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.elements, f, indent=2)

    def dump_to_db(self):
        conn = psycopg.connect(
            dbname="db",
            user="admin",
            password="password",
            host="localhost",
            port=5431
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
                    e["embedding"]
                )
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
        return {
            "type": "table",
            "rows": [[cell.text.strip() for cell in row.cells] for row in table.rows],
        }


    def _extract_paragraph(self, paragraph: Paragraph):
        """
        Parse headers and text body, keeping the current section path up to date.
        """
        style_name = paragraph.style.name.lower()
        if style_name.startswith("heading"):
            try:
                level = int(style_name.replace("heading", "").strip())
            except ValueError:
                level = 1

            # Adjust our current section. 
            while self.current_section and self.current_section[-1].level >= level:
                self.current_section.pop()

            curr_heading = Heading(level, paragraph.text.strip())
            self.current_section.append(curr_heading)
            return {"type": "heading", "text": curr_heading.text, "level": curr_heading.level}
        else:
            return {"type": "paragraph", "text": paragraph.text.strip(), "section_path": [s.text for s in self.current_section]}
        

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
    ex.encode()
    ex.dump_to_file(output_file)
    ex.dump_to_db()

    print(f"Content extracted and saved to: {output_file}")

    


