from dataclasses import dataclass
from typing import List


@dataclass
class Paragraph:
    type: str = "paragraph"
    paragraph_style: str = ""
    text: str = ""
    section_path: List[str] = None
