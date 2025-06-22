from dataclasses import asdict, dataclass
from typing import List, Optional

from .image import Image


@dataclass
class Caption:
    type: str = "caption"
    text: str = ""
    section_path: List[str] = None
    images: Optional[List[Image]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None images."""
        result = asdict(self)
        if result["images"] is None:
            del result["images"]
        return result
