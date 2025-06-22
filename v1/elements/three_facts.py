from dataclasses import dataclass, field
from typing import List

from .document_section import DocumentSection
from .fact import Fact


@dataclass
class ThreeFacts(DocumentSection):
    """Special section for three key facts."""

    type: str = field(default="3_facts", init=False)
    text: str = ""
    section: str = ""
    facts: List[Fact] = None
