from dataclasses import dataclass, field
from typing import Dict

from .document_section import DocumentSection


@dataclass
class Goals2050(DocumentSection):
    """Special section for 2050 goals."""

    type: str = field(default="2050_goals", init=False)
    text: str = ""
    section: str = ""
    values: Dict[str, str] = None
