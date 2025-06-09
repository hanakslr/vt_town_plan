from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

from .action_table import ActionTable
from .caption import Caption
from .goals_2050 import Goals2050
from .heading import Heading
from .paragraph import Paragraph
from .public_engagement import PublicEngagementFindings
from .table import Table
from .three_facts import ThreeFacts

# Union type for all possible document elements
DocumentElement = Union[
    Heading,
    Paragraph,
    Caption,
    ThreeFacts,
    PublicEngagementFindings,
    Goals2050,
    Table,
    ActionTable,
]


def to_dict(obj: Any) -> Dict:
    """Convert an object to a dictionary.

    - Uses custom to_dict() method if available
    - Falls back to dataclasses.asdict() for dataclasses
    - Returns the object itself if it's already a dict
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        return {"value": str(obj)}


class StructuredDocument:
    """
    A structured representation of a chapter document.

    Attributes:
        chapter_number: The chapter number (e.g., "1", "2A")
        title: The chapter title
        goals_2050: The 2050 goals data if present
        three_facts: The three facts block if present
        public_engagement: The public engagement findings if present
        actions: The action table data if present
        content: General content elements (headings, paragraphs, etc.)
    """

    def __init__(
        self,
        chapter_number: Optional[str] = None,
        title: Optional[str] = None,
        goals_2050: Optional[Goals2050] = None,
        three_facts: Optional[ThreeFacts] = None,
        public_engagement: Optional[PublicEngagementFindings] = None,
        actions: Optional[ActionTable] = None,
        content: List[DocumentElement] = None,
    ):
        self.chapter_number = chapter_number
        self.title = title
        self.goals_2050 = goals_2050
        self.three_facts = three_facts
        self.public_engagement = public_engagement
        self.actions = actions
        self.content = content or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the structured document to a dictionary."""
        # Preserve section_path in content items for hierarchical organization
        content_dicts = []
        for item in self.content:
            item_dict = to_dict(item)
            # Ensure section_path is preserved in the output
            if hasattr(item, "section_path") and item.section_path:
                item_dict["section_path"] = item.section_path
            content_dicts.append(item_dict)

        result = {
            "chapter_number": self.chapter_number,
            "title": self.title,
            "content": content_dicts,
        }

        # Only include special sections if they exist
        if self.goals_2050:
            result["2050_goals"] = to_dict(self.goals_2050)

        if self.three_facts:
            result["three_facts"] = to_dict(self.three_facts)

        if self.public_engagement:
            result["public_engagement"] = to_dict(self.public_engagement)

        if self.actions:
            result["actions"] = to_dict(self.actions)

        return result
