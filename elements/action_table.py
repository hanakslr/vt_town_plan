from dataclasses import asdict
from typing import Any, Dict, List

from .objective import Objective
from .strategy import Strategy


class ActionTable:
    """ActionTable class with custom serialization."""

    def __init__(
        self,
        section: str = "",
        objectives: List[Objective] = None,
        strategies: List[Strategy] = None,
        type: str = "action_table",
    ):
        self.type = type
        self.section = section
        self.objectives = objectives or []
        self.strategies = strategies or []

    def __repr__(self):
        return f"ActionTable(section={self.section}, objectives={len(self.objectives)}, strategies={len(self.strategies)})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with proper handling of nested objects."""
        return {
            "type": self.type,
            "section": self.section,
            "objectives": [asdict(obj) for obj in self.objectives],
            "strategies": [strategy.to_dict() for strategy in self.strategies],
        }
