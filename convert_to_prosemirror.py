import glob
import json
import os
from typing import Any, Dict, Optional


class ProseMirrorNode:
    """Base class for all ProSeMirror node types."""

    def __init__(self, node_data: Dict[str, Any]):
        self.type = node_data.get("type", "")
        self.attrs = node_data.get("attrs", {})
        self.content = node_data.get("content", [])
        self._node_data = node_data

    def to_document(self) -> Dict[str, Any]:
        """Convert the node to a LlamaIndex Document."""
        raise NotImplementedError("Subclasses must implement to_document method")

    def get_text(self) -> str:
        """Extract text content from the node."""
        raise NotImplementedError("Subclasses must implement get_text method")


class TextNode(ProseMirrorNode):
    """Represents a text node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.text = node_data.get("text", "")

    def get_text(self) -> str:
        return self.text

    def to_document(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": {"node_type": "text"}}


class HeadingNode(ProseMirrorNode):
    """Represents a heading node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.level = self.attrs.get("level", 1)
        self.text_nodes = [TextNode(c) for c in self.content if c.get("type") == "text"]

    def get_text(self) -> str:
        return " ".join(n.get_text() for n in self.text_nodes)

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {"node_type": "heading", "heading_level": self.level},
        }


class ParagraphNode(ProseMirrorNode):
    """Represents a paragraph node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.text_nodes = [TextNode(c) for c in self.content if c.get("type") == "text"]

    def get_text(self) -> str:
        return " ".join(n.get_text() for n in self.text_nodes)

    def to_document(self) -> Dict[str, Any]:
        return {"text": self.get_text(), "metadata": {"node_type": "paragraph"}}


class GoalsTableNode(ProseMirrorNode):
    """Represents a goals_table node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.text = self.attrs.get("text", "")
        self.section = self.attrs.get("section", "")
        self.values = self.attrs.get("values", {})

    def get_text(self) -> str:
        values_text = "\n".join(
            [f"{key}: {value}" for key, value in self.values.items()]
        )
        return f"{self.text}\nSection: {self.section}\n{values_text}"

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "goals_table",
                "section": self.section,
                "values": self.values,
            },
        }


class ObjectiveNode(ProseMirrorNode):
    """Represents an objective node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")

    def get_text(self) -> str:
        return f"{self.label}: {self.text}"

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {"node_type": "objective", "label": self.label},
        }


class ActionNode(ProseMirrorNode):
    """Represents an action node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")
        self.responsibility = self.attrs.get("responsibility", "")
        self.time_frame = self.attrs.get("time_frame", "")
        self.cost = self.attrs.get("cost", "")

    def get_text(self) -> str:
        return (
            f"{self.label}: {self.text}\n"
            f"Responsibility: {self.responsibility}\n"
            f"Time Frame: {self.time_frame}\n"
            f"Cost: {self.cost}"
        )

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "action",
                "label": self.label,
                "responsibility": self.responsibility,
                "time_frame": self.time_frame,
                "cost": self.cost,
            },
        }


class StrategyNode(ProseMirrorNode):
    """Represents a strategy node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")
        self.actions = [
            ActionNode(a) for a in self.content if a.get("type") == "action"
        ]

    def get_text(self) -> str:
        return f"{self.label}: {self.text}"

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {"node_type": "strategy", "label": self.label},
        }


class ActionTableNode(ProseMirrorNode):
    """Represents an action_table node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.objectives = [
            ObjectiveNode(n) for n in self.content if n.get("type") == "objective"
        ]
        self.strategies = [
            StrategyNode(n) for n in self.content if n.get("type") == "strategy"
        ]

    def get_text(self) -> str:
        obj_texts = [obj.get_text() for obj in self.objectives]
        strat_texts = [strat.get_text() for strat in self.strategies]
        return "\n\n".join(obj_texts + strat_texts)

    def to_document(self) -> Dict[str, Any]:
        return {"text": self.get_text(), "metadata": {"node_type": "action_table"}}


class DocNode(ProseMirrorNode):
    """Represents the root doc node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        super().__init__(node_data)
        self.chapter = self.attrs.get("chapter", "")
        self.nodes = []
        self.title = ""

        for item in self.content:
            node_type = item.get("type", "")
            if node_type == "heading":
                heading_node = HeadingNode(item)
                # Save the first level 1 heading as the title
                if heading_node.level == 1 and not self.title:
                    self.title = heading_node.get_text()
                self.nodes.append(heading_node)
            elif node_type == "paragraph":
                self.nodes.append(ParagraphNode(item))
            elif node_type == "goals_table":
                self.nodes.append(GoalsTableNode(item))
            elif node_type == "action_table":
                self.nodes.append(ActionTableNode(item))

    def get_text(self) -> str:
        return "\n\n".join([node.get_text() for node in self.nodes])

    def get_title(self) -> str:
        """Get the chapter title from the first level 1 heading."""
        return self.title

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "doc",
                "chapter": self.chapter,
                "title": self.title,
            },
        }


def create_node_from_data(node_data: Dict[str, Any]) -> Optional[ProseMirrorNode]:
    """Create the appropriate node type from the given data."""
    node_type = node_data.get("type", "")

    if node_type == "doc":
        return DocNode(node_data)
    elif node_type == "heading":
        return HeadingNode(node_data)
    elif node_type == "paragraph":
        return ParagraphNode(node_data)
    elif node_type == "text":
        return TextNode(node_data)
    elif node_type == "goals_table":
        return GoalsTableNode(node_data)
    elif node_type == "objective":
        return ObjectiveNode(node_data)
    elif node_type == "strategy":
        return StrategyNode(node_data)
    elif node_type == "action":
        return ActionNode(node_data)
    elif node_type == "action_table":
        return ActionTableNode(node_data)
    return None


def convert_block(block):
    if block["type"] == "heading":
        return {
            "type": "heading",
            "attrs": {"level": block["level"]},
            "content": [{"type": "text", "text": block["text"]}],
        }
    elif block["type"] == "paragraph":
        return {
            "type": "paragraph",
            "content": [{"type": "text", "text": block["text"]}],
        }
    return None


def convert_goals(goals):
    return {
        "type": "goals_table",
        "attrs": {
            "text": goals["text"],
            "section": goals["section"],
            "values": goals["values"],
        },
    }


def convert_objective(obj):
    return {"type": "objective", "attrs": {"label": obj["label"], "text": obj["text"]}}


def convert_action(action):
    return {
        "type": "action",
        "attrs": {
            "label": action["label"],
            "text": action["text"],
            "responsibility": action.get("responsibility", ""),
            "time_frame": action.get("time_frame", ""),
            "cost": action.get("cost", ""),
        },
    }


def convert_strategy(strategy):
    return {
        "type": "strategy",
        "attrs": {"label": strategy["label"], "text": strategy["text"]},
        "content": [convert_action(a) for a in strategy.get("actions", [])],
    }


def convert_actions_table(actions):
    content = []

    for obj in actions.get("objectives", []):
        content.append(convert_objective(obj))

    for strat in actions.get("strategies", []):
        content.append(convert_strategy(strat))

    return {"type": "action_table", "content": content}


def convert_document(custom_json):
    doc = {
        "type": "doc",
        "content": [],
        "attrs": {"chapter": custom_json.get("chapter_number")},
    }

    # Add a chapter title first
    title_block = {
        "type": "heading",
        "attrs": {"level": 1},
        "content": [{"type": "text", "text": custom_json.get("title")}],
    }

    doc["content"].append(title_block)

    for item in custom_json.get("content", []):
        block = convert_block(item)
        if block:
            doc["content"].append(block)

    if "2050_goals" in custom_json:
        doc["content"].append(convert_goals(custom_json["2050_goals"]))

    if "actions" in custom_json:
        doc["content"].append(convert_actions_table(custom_json["actions"]))

    return doc


def convert_file(input_file, output_file):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            custom_json = json.load(f)

        prosemirror_doc = convert_document(custom_json)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(prosemirror_doc, f, indent=2, ensure_ascii=False)

        print(f"Converted {os.path.basename(input_file)} to {output_file}")
        return True
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False


def main():
    input_dir = "output/chapters"
    output_dir = "output/chapters/prosemirror"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the input directory
    input_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not input_files:
        print(f"No JSON files found in {input_dir}")
        return

    successful = 0
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        if convert_file(input_file, output_file):
            successful += 1

    print(
        f"Conversion complete: {successful}/{len(input_files)} files converted successfully"
    )


if __name__ == "__main__":
    main()
