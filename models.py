from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class Heading:
    type: str = "heading"
    level: int = 1
    text: str = ""
    chapter_number: Optional[str] = None
    section: Optional[str] = None


@dataclass
class Paragraph:
    type: str = "paragraph"
    paragraph_style: str = ""
    text: str = ""
    section_path: List[str] = None


@dataclass
class Caption:
    type: str = "caption"
    text: str = ""
    section_path: List[str] = None


@dataclass
class Fact:
    title: str
    text: str


@dataclass
class ThreeFacts:
    type: str = "3_facts"
    text: str = ""
    section: str = ""
    facts: List[Fact] = None


@dataclass
class PublicEngagementFindings:
    type: str = "3_public_engagement_findings"
    text: str = ""
    section: str = ""
    facts: List[Fact] = None


@dataclass
class Goals2050:
    type: str = "2050_goals"
    text: str = ""
    section: str = ""
    values: Dict[str, str] = None


@dataclass
class Table:
    type: str = "table"
    rows: List[List[str]] = None
    border_info: Optional[Dict[str, Any]] = None
    border_color: Optional[str] = None


class Action:
    """Action class with custom serialization behavior."""

    def __init__(
        self,
        label: str,
        text: str,
        responsibility: str,
        time_frame: str,
        cost: str,
        starred: Optional[bool] = None,
        multiple_strategies: Optional[bool] = None,
    ):
        self.label = label
        self.text = text
        self.responsibility = responsibility
        self.time_frame = time_frame
        self.cost = cost
        self.starred = starred
        self.multiple_strategies = multiple_strategies

    def __repr__(self):
        return f"Action(label={self.label}, text={self.text})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None/False values for optional fields."""
        result = {
            "label": self.label,
            "text": self.text,
            "responsibility": self.responsibility,
            "time_frame": self.time_frame,
            "cost": self.cost,
        }

        # Only include starred and multiple_strategies if they're True
        if self.starred:
            result["starred"] = True

        if self.multiple_strategies:
            result["multiple_strategies"] = True

        return result


class Strategy:
    """Strategy class with custom serialization for actions."""

    def __init__(self, label: str, text: str, actions: List[Action]):
        self.label = label
        self.text = text
        self.actions = actions

    def __repr__(self):
        return f"Strategy(label={self.label}, text={self.text}, actions={len(self.actions)})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with proper handling of Action objects."""
        return {
            "label": self.label,
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class Objective:
    label: str
    text: str


class ActionTable:
    """ActionTable class with custom serialization."""

    def __init__(
        self,
        section: str = "",
        objectives: List[Objective] = None,
        strategies: List[Strategy] = None,
        type: str = "action_table",
    ):
        self.type = type
        self.section = section
        self.objectives = objectives or []
        self.strategies = strategies or []

    def __repr__(self):
        return f"ActionTable(section={self.section}, objectives={len(self.objectives)}, strategies={len(self.strategies)})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with proper handling of nested objects."""
        return {
            "type": self.type,
            "section": self.section,
            "objectives": [asdict(obj) for obj in self.objectives],
            "strategies": [strategy.to_dict() for strategy in self.strategies],
        }


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
