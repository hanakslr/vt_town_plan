from dataclasses import dataclass
from typing import Dict


@dataclass
class Goals2050:
    type: str = "2050_goals"
    text: str = ""
    section: str = ""
    values: Dict[str, str] = None
