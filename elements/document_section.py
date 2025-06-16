from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class DocumentSection(ABC):
    """Base class for all special document sections.

    This class provides a common interface and default implementation for special
    sections in the document like Goals2050, ThreeFacts, etc.

    Attributes:
        type: The type identifier for this section
        text: Optional descriptive text for the section
        section: Optional section identifier
    """

    type: str
    text: str = ""
    section: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the section to a dictionary.

        By default, uses dataclasses.asdict() to serialize the object.
        Subclasses can override this method to provide custom serialization.
        """
        return asdict(self)
