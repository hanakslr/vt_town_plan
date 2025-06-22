from dataclasses import dataclass, field
from typing import List

from .document_section import DocumentSection
from .fact import Fact


@dataclass
class PublicEngagementFindings(DocumentSection):
    """Special section for public engagement findings."""

    type: str = field(default="3_public_engagement_findings", init=False)
    facts: List[Fact] = None
