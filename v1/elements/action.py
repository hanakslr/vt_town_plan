from typing import Any, Dict, Optional


class Action:
    """Action class with custom serialization behavior."""

    def __init__(
        self,
        label: str,
        text: str,
        responsibility: str,
        time_frame: str,
        cost: str,
        starred: Optional[bool] = None,
        multiple_strategies: Optional[bool] = None,
    ):
        self.label = label
        self.text = text
        self.responsibility = responsibility
        self.time_frame = time_frame
        self.cost = cost
        self.starred = starred
        self.multiple_strategies = multiple_strategies

    def __repr__(self):
        return f"Action(label={self.label}, text={self.text})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None/False values for optional fields."""
        result = {
            "label": self.label,
            "text": self.text,
            "responsibility": self.responsibility,
            "time_frame": self.time_frame,
            "cost": self.cost,
        }

        # Only include starred and multiple_strategies if they're True
        if self.starred:
            result["starred"] = True

        if self.multiple_strategies:
            result["multiple_strategies"] = True

        return result
