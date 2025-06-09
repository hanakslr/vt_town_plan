from typing import Any, Dict, List

from .action import Action


class Strategy:
    """Strategy class with custom serialization for actions."""

    def __init__(self, label: str, text: str, actions: List[Action]):
        self.label = label
        self.text = text
        self.actions = actions

    def __repr__(self):
        return f"Strategy(label={self.label}, text={self.text}, actions={len(self.actions)})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with proper handling of Action objects."""
        return {
            "label": self.label,
            "text": self.text,
            "actions": [action.to_dict() for action in self.actions],
        }
