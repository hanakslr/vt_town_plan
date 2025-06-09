from dataclasses import dataclass
from typing import Optional


@dataclass
class Heading:
    type: str = "heading"
    level: int = 1
    text: str = ""
    chapter_number: Optional[str] = None
    section: Optional[str] = None
