from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Image:
    """Represents an image extracted from the document.

    Attributes:
        filename: The name of the image file
        alt_text: Optional alternative text for the image
        section_path: Optional list of section headings that contain this image
        width: Optional width of the image in pixels
        height: Optional height of the image in pixels
    """

    filename: str
    alt_text: Optional[str] = None
    section_path: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None
