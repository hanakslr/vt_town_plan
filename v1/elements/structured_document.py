from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from .action_table import ActionTable
from .caption import Caption
from .document_section import DocumentSection
from .goals_2050 import Goals2050
from .image import Image
from .paragraph import Paragraph
from .public_engagement import PublicEngagementFindings
from .three_facts import ThreeFacts


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
        images: List of all images in the document
    """

    def __init__(
        self,
        chapter_number: Optional[str] = None,
        title: Optional[str] = None,
        goals_2050: Optional[Goals2050] = None,
        three_facts: Optional[ThreeFacts] = None,
        public_engagement: Optional[PublicEngagementFindings] = None,
        actions: Optional[ActionTable] = None,
        content: List[DocumentSection] = None,
        images: List[Image] = None,
    ):
        self.chapter_number = chapter_number
        self.title = title
        self.goals_2050 = goals_2050
        self.three_facts = three_facts
        self.public_engagement = public_engagement
        self.actions = actions
        self.content = content or []
        self.images = images or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the structured document to a dictionary."""
        # Preserve section_path in content items for hierarchical organization
        content_dicts = []
        for item in self.content:
            if isinstance(item, (Paragraph, Caption)):
                item_dict = item.to_dict()
            elif isinstance(item, DocumentSection):
                item_dict = item.to_dict()
            elif is_dataclass(item):
                item_dict = asdict(item)
            else:
                item_dict = {"value": str(item)}

            # Ensure section_path is preserved in the output
            if hasattr(item, "section_path") and item.section_path:
                item_dict["section_path"] = item.section_path
            content_dicts.append(item_dict)

        # Convert images to dictionaries, handling both dataclass and non-dataclass instances
        image_dicts = []
        for img in self.images:
            if is_dataclass(img):
                image_dicts.append(asdict(img))
            elif isinstance(img, dict):
                image_dicts.append(img)
            else:
                image_dicts.append({"value": str(img)})

        result = {
            "chapter_number": self.chapter_number,
            "title": self.title,
            "content": content_dicts,
        }

        if image_dicts:
            result["images"] = image_dicts

        # Only include special sections if they exist
        if self.goals_2050:
            result["2050_goals"] = self.goals_2050.to_dict()

        if self.three_facts:
            result["three_facts"] = self.three_facts.to_dict()

        if self.public_engagement:
            result["public_engagement"] = self.public_engagement.to_dict()

        if self.actions:
            result["actions"] = self.actions.to_dict()

        return result
