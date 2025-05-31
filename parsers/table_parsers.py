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
        """Extract rows and styles from the table."""
        for row_idx, row in enumerate(self.table.rows):
            cols = []
            for j, cell in enumerate(row.cells):
                cell_id = id(cell._tc)
                if cell_id in self.seen_cells:
                    continue
                self.seen_cells.add(cell_id)

                # Handle merged cells
                is_vertical_merge = self._is_merged_vertically(cell._tc)
                is_horizontal_merge = self._is_merged_horizontally(cell._tc)
                grid_span = self._get_grid_span(cell._tc) if is_horizontal_merge else 1
                is_origin = self._is_merge_origin(cell._tc)
                has_content = bool(cell.text.strip())

                # Debug info for objectives
                if cell.text.strip().startswith("5."):
                    print(f"DEBUG: Found objective cell: '{cell.text.strip()}'")
                    print(f"  Vertical merge: {is_vertical_merge}")
                    print(f"  Horizontal merge: {is_horizontal_merge}")
                    print(f"  Grid span: {grid_span}")
                    print(f"  Is origin: {is_origin}")

                # Skip cells that are continuations of a merge AND have no content
                if is_vertical_merge and not is_origin and not has_content:
                    continue

                list_num = None
                style_info = None

                # Process paragraphs in the cell
                for para in cell.paragraphs:
                    # Extract list numbers if applicable
                    if self.list_number_generator:
                        list_num = self._extract_list_number(para)

                    # Extract style information
                    style_info = self._extract_style_info(para)

                text = cell.text.strip() or list_num

                if text:
                    # Detect if this cell contains an objective label (e.g., "5.A", "3.B", etc.)
                    # Pattern: digit + dot + uppercase letter
                    import re
                    is_objective_label = bool(re.match(r'^\d+\.[A-Z]$', text.strip()))
                    
                    # Avoid adding duplicate text
                    if text not in cols:
                        # Handle objective label cells specially
                        if is_objective_label:
                            # This is an objective label, add it first
                            cols.append(text)
                            
                            # Find the description in the next cell
                            next_cell_idx = j + 1  # Use the actual cell index instead of column count
                            if next_cell_idx < len(row.cells):
                                next_cell = row.cells[next_cell_idx]
                                next_text = next_cell.text.strip()
                                if next_text and next_text not in cols:
                                    # Add the description and mark the cell as seen
                                    cols.append(next_text)
                                    self.seen_cells.add(id(next_cell._tc))
                        # Handle horizontally merged cells
                        elif is_horizontal_merge:
                            # Add the content of the merged cell
                            cols.append(text)
                        else:
                            # Normal cell
                            cols.append(text)

                    if style_info:
                        self.styles.append(style_info)

            if cols:
                self.rows.append(cols)

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
        """
        Create output dictionary from parsed data. This is the base class implementation.
        Tables with special handling are done so in the approriate child class.
        """
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
        return {
            "type": "heading",
            "level": 1,
            "text": self.rows[0][1],
            "chaper_number": self.rows[0][0],
        }

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

        # if rows[0][0] == "Objectives, Strategies, and Actions":
        #     return True

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
        return (objective_count > 0 and strategy_count > 0 and action_count > 0) or (
            strategy_count >= 2 and action_count >= 3
        )


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
