from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Table:
    type: str = "table"
    rows: List[List[str]] = None
    border_info: Optional[Dict[str, Any]] = None
    border_color: Optional[str] = None
