from dataclasses import dataclass
from typing import List


@dataclass
class Caption:
    type: str = "caption"
    text: str = ""
    section_path: List[str] = None
