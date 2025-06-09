from dataclasses import dataclass
from typing import List

from .fact import Fact


@dataclass
class ThreeFacts:
    type: str = "3_facts"
    text: str = ""
    section: str = ""
    facts: List[Fact] = None
