"""
Table parsers for different types of tables in the document.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Dict, List, Optional

from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


class TableType(Enum):
    """Enum for different types of tables in the document."""

    GENERIC = auto()
    CHAPTER_HEADER = auto()
    GOALS_2050 = auto()
    THREE_FACTS = auto()
    THREE_FINDINGS = auto()
    ACTION_TABLE = auto()


@dataclass
class TableStyles:
    """Styles extracted from table cells."""

    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    text: Optional[str] = None


def extract_table_borders(table: Table) -> Dict:
    """
    Extract border style information from a table.

    Returns a dictionary with border attributes.
    """
    border_info = {}

    # Access the underlying XML element
    tbl_element = table._element

    # Look for tblPr (table properties)
    tbl_pr = tbl_element.find(
        ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblPr"
    )
    if tbl_pr is not None:
        # Look for table borders
        tbl_borders = tbl_pr.find(
            ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tblBorders"
        )
        if tbl_borders is not None:
            # Extract border colors from all sides
            for border_type in ["top", "left", "bottom", "right", "insideH", "insideV"]:
                border = tbl_borders.find(
                    f".//{{{tbl_element.nsmap['w']}}}{border_type}"
                )
                if border is not None:
                    # Get color attribute
                    color = border.get(f"{{{tbl_element.nsmap['w']}}}color")
                    if color:
                        border_info[f"{border_type}_color"] = color

                    # Get border style (single, double, etc.)
                    val = border.get(f"{{{tbl_element.nsmap['w']}}}val")
                    if val:
                        border_info[f"{border_type}_style"] = val

                    # Get border width
                    sz = border.get(f"{{{tbl_element.nsmap['w']}}}sz")
                    if sz:
                        border_info[f"{border_type}_width"] = sz

    return border_info


@dataclass
class BaseTableParser:
    """Base class for all table parsers."""

    table: Table
    current_section: List[str]
    list_number_generator: Optional[Any] = None

    # Class variables for type checking
    table_type: ClassVar[TableType] = TableType.GENERIC

    def __post_init__(self):
        self.seen_cells = set()
        self.rows = []
        self.styles = []
        self.cell_texts = []
        self.border_info = extract_table_borders(self.table)

    def parse(self) -> Dict:
        """Parse the table and return structured data."""
        self._extract_rows_and_styles()

        if not self.rows:
            return None

        return self._create_output()

    def _extract_rows_and_styles(self):
        """
        Extract rows and styles from the table using a two-pass approach for more robustness.
        """
        # First pass: collect all cell information without any filtering
        # This ensures we have complete information before making decisions
        raw_rows = []

        for row in self.table.rows:
            raw_cells = []
            for cell in row.cells:
                # Check for merged cells
                is_vertical_merge = self._is_merged_vertically(cell._tc)
                is_horizontal_merge = self._is_merged_horizontally(cell._tc)
                grid_span = self._get_grid_span(cell._tc) if is_horizontal_merge else 1
                is_origin = self._is_merge_origin(cell._tc)

                # Get text content
                text = cell.text.strip()
                list_num = None

                # Get style information
                style_info = None
                for para in cell.paragraphs:
                    # Extract list numbers if applicable
                    if self.list_number_generator:
                        list_num = self._extract_list_number(para)
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

                # Store all information for this cell
                raw_cells.append(
                    {
                        "cell": cell,
                        "text": text or list_num,
                        "style": style_info,
                        "cell_id": id(cell._tc),
                        "is_vertical_merge": is_vertical_merge,
                        "is_horizontal_merge": is_horizontal_merge,
                        "grid_span": grid_span,
                        "is_origin": is_origin,
                    }
                )

                # if (
                #     "Develop a communication schedule to proactively share data, such"
                #     in text
                # ):
                #     print(raw_cells)
                #     for cell_dict in raw_cells:
                #         cell = cell_dict["cell"]
                #         xml = (
                #             cell._tc
                #         )  # _tc is the lxml element representing the <w:tc> tag
                #         print(
                #             etree.tostring(xml, pretty_print=True, encoding="unicode")
                #         )

                #     raise "stop"

            raw_rows.append(raw_cells)

        # Second pass: process rows with full contextual information
        # Since we have all the info up front, we can make better decisions
        for row_idx, row_cells in enumerate(raw_rows):
            cols = []
            seen_in_this_row = set()  # Track processed cells in this row only

            # Process each cell in the row
            for cell_idx, cell_info in enumerate(row_cells):
                # Skip if already processed in this row
                if cell_info["cell_id"] in seen_in_this_row:
                    continue

                seen_in_this_row.add(cell_info["cell_id"])

                # Skip cells that are continuations of vertical merges and have no content
                if (
                    cell_info["is_vertical_merge"]
                    and not cell_info["is_origin"]
                    and not cell_info["text"]
                ):
                    continue

                text = cell_info["text"]
                if not text:
                    continue

                # # Important: For the first cell in a row, check if it might be a label
                # # with a description in the next cell
                # if cell_idx == 0 and len(row_cells) > 1:
                #     # Try to detect if this is a label-description pattern
                #     # Typically, labels are short and in the first column
                #     is_label = len(text) <= 8  # Generous limit for a label

                #     if is_label:
                #         # Add the label first
                #         cols.append(text)

                #         # Look for description in the next cell
                #         if cell_idx + 1 < len(row_cells):
                #             next_cell = row_cells[cell_idx + 1]
                #             if next_cell["text"] and next_cell["text"] not in cols:
                #                 # Add the description
                #                 cols.append(next_cell["text"])
                #                 # Mark as processed
                #                 seen_in_this_row.add(next_cell["cell_id"])

                #                 # Also add to the overall seen cells
                #                 self.seen_cells.add(next_cell["cell_id"])

                #         # Save style info
                #         if cell_info["style"]:
                #             self.styles.append(cell_info["style"])

                #         continue  # Move to the next unprocessed cell

                # For other cells, just add the text if not a duplicate
                if text not in cols:
                    cols.append(text)

                # Save style info
                if cell_info["style"]:
                    self.styles.append(cell_info["style"])

            # Add the processed row if it has content
            if cols:
                self.rows.append(cols)

            # Add all cell IDs to the overall seen set to avoid reprocessing
            for cell_info in row_cells:
                self.seen_cells.add(cell_info["cell_id"])

    def _is_merged_vertically(self, tc):
        """Check if a cell is merged vertically."""
        vMerge = tc.find(".//w:vMerge", tc.nsmap)
        if vMerge is not None:
            val = vMerge.get(qn("w:val"))
            return val == "continue" or val is None  # continuation
        return False

    def _is_merged_horizontally(self, tc):
        """Check if a cell is merged horizontally."""
        gridSpan = tc.find(".//w:gridSpan", tc.nsmap)
        return gridSpan is not None

    def _get_grid_span(self, tc):
        """Get the grid span value for a horizontally merged cell."""
        gridSpan = tc.find(".//w:gridSpan", tc.nsmap)
        if gridSpan is not None:
            return int(gridSpan.get(qn("w:val"), "1"))
        return 1

    def _is_merge_origin(self, tc):
        """Check if a cell is the origin of a merge."""
        vMerge = tc.find(".//w:vMerge", tc.nsmap)
        return vMerge is None or vMerge.get(qn("w:val")) == "restart"

    def _extract_list_number(self, para: Paragraph) -> Optional[str]:
        """Extract list number from paragraph if present."""
        if not self.list_number_generator:
            return None

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
                return self.list_number_generator(numId, ilvl)
        return None

    def _extract_style_info(self, para: Paragraph) -> Optional[TableStyles]:
        """Extract style information from paragraph."""
        for run in para.runs:
            font = run.font
            return TableStyles(
                text=run.text,
                font_name=font.name,
                font_size=font.size.pt if font.size else None,
                bold=font.bold,
                italic=font.italic,
            )
        return None

    def _create_output(self) -> Dict:
        """Create output dictionary from parsed data."""
        result = {
            "type": "table",
            "rows": self.rows,
        }

        # Include border information in the output if available
        if self.border_info:
            result["border_info"] = self.border_info

            # For convenience, add a simplified border_color field for quick comparison
            # Use the top border color as the representative color
            if "top_color" in self.border_info:
                result["border_color"] = self.border_info["top_color"]

        return result

    @classmethod
    def can_parse(
        cls, table: Table, styles: List[TableStyles], rows: List[List[str]]
    ) -> bool:
        """Determine if this parser can handle the table."""
        return True  # Base parser can handle any table


@dataclass
class ChapterHeaderParser(BaseTableParser):
    """Parser for chapter header tables."""

    table_type: ClassVar[TableType] = TableType.CHAPTER_HEADER

    def _create_output(self) -> Dict:
        """Create output for chapter header table."""
        return {"type": "heading", "level": 1, "text": self.rows[0][1]}

    @classmethod
    def can_parse(
        cls, table: Table, styles: List[TableStyles], rows: List[List[str]]
    ) -> bool:
        """Determine if this parser can handle the table."""
        # Check if this is a chapter header table
        return (
            len(rows) == 1
            and len(rows[0]) == 2
            and all(
                [
                    s.font_name == "Bumper Sticker" and float(s.font_size) >= 26.0
                    for s in styles
                    if s.font_name and s.font_size
                ]
            )
        )


@dataclass
class GoalsTableParser(BaseTableParser):
    """Parser for 2050 goals tables."""

    table_type: ClassVar[TableType] = TableType.GOALS_2050

    def _create_output(self) -> Dict:
        """Create output for 2050 goals table."""
        from parsers.goals_table_parser import parse_goals_table

        values = parse_goals_table(self.rows)

        return {
            "type": "2050_goals",
            "text": self.rows[0][0],
            "section": self.current_section[0],
            "values": values,
        }

    @classmethod
    def can_parse(
        cls, table: Table, styles: List[TableStyles], rows: List[List[str]]
    ) -> bool:
        """Determine if this parser can handle the table."""
        return rows and rows[0] and rows[0][0].startswith("Goals: In 2050")


@dataclass
class FactsTableParser(BaseTableParser):
    """Parser for Three Things tables."""

    table_type: ClassVar[TableType] = TableType.THREE_FACTS

    def _create_output(self) -> Dict:
        """Create output for Three Things table."""
        from parsers.facts_table_parser import parse_facts_table

        facts = parse_facts_table(self.rows)

        return {
            "type": "3_public_engagement_findings"
            if "Public Engagement" in self.rows[0][0]
            else "3_facts",
            "text": self.rows[0][0],
            "section": self.current_section[0],
            "facts": facts,
        }

    @classmethod
    def can_parse(
        cls, table: Table, styles: List[TableStyles], rows: List[List[str]]
    ) -> bool:
        """Determine if this parser can handle the table."""
        return rows and rows[0] and rows[0][0].startswith("Three Things")


@dataclass
class ActionTableParser(BaseTableParser):
    """Parser for action tables."""

    table_type: ClassVar[TableType] = TableType.ACTION_TABLE

    def _create_output(self) -> Dict:
        """
        Create output for action table.

        Structure:
        {
            "type": "action_table",
            "section": "Arts and social infrastructure",
            "objectives": [
                {"label": "2.A", "text": "...."}
            ],
            "strategies": [
                {
                    "label": "2.1",
                    "text": "...",
                    "actions": [
                        {
                            "label": "2.1.1",
                            "text": "...",
                            "responsibility": "...",
                            "time_frame": "...",
                            "cost": "...",
                            "starred": true,
                            "multiple_strategies": true
                        }
                    ]
                }
            ]
        }
        """
        from parsers.action_table_parser import parse_action_table

        parsed_data = parse_action_table(self.rows)

        return {
            "type": "action_table",
            "section": self.current_section[0] if self.current_section else "",
            "objectives": parsed_data["objectives"],
            "strategies": parsed_data["strategies"],
        }

    @classmethod
    def can_parse(
        cls, table: Table, styles: List[TableStyles], rows: List[List[str]]
    ) -> bool:
        """Determine if this parser can handle the table."""
        import re

        # Check for patterns that indicate this is an action table
        if not rows:
            return False

        if rows[0][0] == "Objectives, Strategies, and Actions":
            return True

        # Look for patterns like "2.A", "2.1", "2.1.1" in the first column
        objective_pattern = re.compile(r"^\d+\.[A-Z]")
        strategy_pattern = re.compile(r"^\d+\.\d+")
        action_pattern = re.compile(r"^\d+\.\d+\.\d+")

        # Count matches for each pattern
        objective_count = 0
        strategy_count = 0
        action_count = 0

        for row in rows:
            if not row:
                continue

            cell_text = row[0]

            if objective_pattern.match(cell_text):
                objective_count += 1
            elif strategy_pattern.match(cell_text):
                strategy_count += 1
            elif action_pattern.match(cell_text):
                action_count += 1

        # If we have at least one of each pattern, or a significant number of actions
        # and strategies, this is likely an action table

        # Right now actions don't get their numbers until they are in the parser??
        # so after this. Not sure how strategies are getting their numbers now.
        return strategy_count >= 2 or action_count >= 2 or objective_count >= 2


class TableParserFactory:
    """Factory for creating table parsers."""

    # Order matters - more specific parsers should come first
    parser_classes = [
        ChapterHeaderParser,
        GoalsTableParser,
        FactsTableParser,
        ActionTableParser,
        BaseTableParser,  # Fallback parser
    ]

    @classmethod
    def create_parser(
        cls,
        table: Table,
        rows: List[List[str]],
        styles: List[TableStyles],
        current_section: List[str],
        list_number_generator=None,
    ) -> BaseTableParser:
        """Create appropriate parser for the table."""
        for parser_class in cls.parser_classes:
            if parser_class.can_parse(table, styles, rows):
                return parser_class(
                    table=table,
                    current_section=current_section,
                    list_number_generator=list_number_generator,
                )

        # Fallback to generic table parser
        return BaseTableParser(
            table=table,
            current_section=current_section,
            list_number_generator=list_number_generator,
        )


class TableMerger:
    """Utility for merging tables that are split across page breaks."""

    @staticmethod
    def should_merge(prev_table: Dict, current_table: Dict) -> bool:
        """Determine if two tables should be merged."""
        # Tables must both be generic tables
        if prev_table.get("type") != "table" or current_table.get("type") != "table":
            return False

        # Check for other criteria that would indicate these are parts of the same table
        prev_rows = prev_table.get("rows", [])
        curr_rows = current_table.get("rows", [])

        if not prev_rows or not curr_rows:
            return False

        # Check if number of columns matches in the last row of prev table and first row of current table
        if len(prev_rows[-1]) != len(curr_rows[0]):
            return False

        # Compare border colors - this is crucial for determining if tables should be merged
        prev_color = prev_table.get("border_color")
        curr_color = current_table.get("border_color")

        # If both tables have border colors and they don't match, don't merge
        if prev_color and curr_color and prev_color != curr_color:
            return False

        # Check border styles too if available
        prev_border_info = prev_table.get("border_info", {})
        curr_border_info = current_table.get("border_info", {})

        # Check a few key border attributes
        for attr in ["top_style", "left_style", "bottom_style", "right_style"]:
            prev_style = prev_border_info.get(attr)
            curr_style = curr_border_info.get(attr)
            if prev_style and curr_style and prev_style != curr_style:
                return False

        # The tables have matching structure and styles, so they should be merged
        return True

    @staticmethod
    def merge_tables(prev_table: Dict, current_table: Dict) -> Dict:
        """Merge two tables."""
        merged_rows = prev_table.get("rows", []) + current_table.get("rows", [])

        # Create merged result
        result = {
            "type": "table",
            "rows": merged_rows,
        }

        # Preserve metadata from the first table
        for key in prev_table:
            if key not in ["type", "rows"] and key not in result:
                result[key] = prev_table[key]

        return result
