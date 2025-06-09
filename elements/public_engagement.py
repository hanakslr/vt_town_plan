from dataclasses import dataclass
from typing import List

from .fact import Fact


@dataclass
class PublicEngagementFindings:
    type: str = "3_public_engagement_findings"
    text: str = ""
    section: str = ""
    facts: List[Fact] = None
