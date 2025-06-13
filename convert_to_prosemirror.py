import glob
import json
import os
from typing import Any, Dict, List, Optional


class ProseMirrorNode:
    """Base class for all ProSeMirror node types."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        self.type = node_data.get("type", "")
        self.attrs = node_data.get(
            "attrs", {}
        ).copy()  # Create a copy to avoid modifying the original
        self.content = node_data.get("content", [])
        self._node_data = node_data
        self.index = index
        self.chapter = chapter

        # Generate a unique ID if not already present
        if "id" not in self.attrs:
            self.attrs["id"] = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique, deterministic ID for this node."""
        # Default implementation, should be overridden by subclasses
        chapter_letters = "".join(c for c in self.chapter if c.isalpha())
        return f"c{chapter_letters}-{self.type}{self.index}"

    def to_document(self) -> Dict[str, Any]:
        """Convert the node to a LlamaIndex Document."""
        raise NotImplementedError("Subclasses must implement to_document method")

    def get_text(self) -> str:
        """Extract text content from the node."""
        raise NotImplementedError("Subclasses must implement get_text method")

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for JSON serialization."""
        result = {"type": self.type, "attrs": self.attrs}

        if self.content:
            if isinstance(self.content[0], ProseMirrorNode):
                result["content"] = [node.to_dict() for node in self.content]
            else:
                result["content"] = self.content

        return result


class TextNode(ProseMirrorNode):
    """Represents a text node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.text = node_data.get("text", "")

    def get_text(self) -> str:
        return self.text

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": {"node_type": "text", "id": self.attrs.get("id", "")},
        }


class HeadingNode(ProseMirrorNode):
    """Represents a heading node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.level = self.attrs.get("level", 1)
        self.text_nodes = [
            TextNode(c, i, chapter)
            for i, c in enumerate(self.content)
            if c.get("type") == "text"
        ]

    def get_text(self) -> str:
        return " ".join(n.get_text() for n in self.text_nodes)

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "heading",
                "heading_level": self.level,
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        # Generate a heading ID based on level and text
        text = self.get_text()
        slug = text.lower().replace(" ", "-")[:8]  # Use first 8 chars of slug
        return f"c{self.chapter}-h{self.level}{self.index}-{slug}"


class ParagraphNode(ProseMirrorNode):
    """Represents a paragraph node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.text_nodes = [
            TextNode(c, i, chapter)
            for i, c in enumerate(self.content)
            if c.get("type") == "text"
        ]

    def get_text(self) -> str:
        return " ".join(n.get_text() for n in self.text_nodes)

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {"node_type": "paragraph", "id": self.attrs.get("id", "")},
        }

    def _generate_id(self) -> str:
        return f"c{self.chapter}-p{self.index}"


class GoalsTableNode(ProseMirrorNode):
    """Represents a goals_table node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
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
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        return f"c{self.chapter}-goals"


class ObjectiveNode(ProseMirrorNode):
    """Represents an objective node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")

    def get_text(self) -> str:
        return f"{self.label}: {self.text}"

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "objective",
                "label": self.label,
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        return f"c{self.chapter}-obj-{self.label}"


class ActionNode(ProseMirrorNode):
    """Represents an action node in ProSeMirror."""

    def __init__(
        self,
        node_data: Dict[str, Any],
        index: int = 0,
        chapter: str = "",
        strategy_label: str = "",
    ):
        super().__init__(node_data, index, chapter)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")
        self.responsibility = self.attrs.get("responsibility", "")
        self.time_frame = self.attrs.get("time_frame", "")
        self.cost = self.attrs.get("cost", "")
        self.strategy_label = strategy_label

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
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        if self.strategy_label:
            return f"{self.chapter}-action-{self.strategy_label}-{self.label}"
        return f"c{self.chapter}-action-{self.label}"


class StrategyNode(ProseMirrorNode):
    """Represents a strategy node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.label = self.attrs.get("label", "")
        self.text = self.attrs.get("text", "")
        self.actions = [
            ActionNode(a, i, chapter, self.label)
            for i, a in enumerate(self.content)
            if a.get("type") == "action"
        ]

    def get_text(self) -> str:
        return f"{self.label}: {self.text}"

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {
                "node_type": "strategy",
                "label": self.label,
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        return f"c{self.chapter}-strategy-{self.label}"


class ActionTableNode(ProseMirrorNode):
    """Represents an action_table node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any], index: int = 0, chapter: str = ""):
        super().__init__(node_data, index, chapter)
        self.objectives = [
            ObjectiveNode(n, i, chapter)
            for i, n in enumerate(self.content)
            if n.get("type") == "objective"
        ]
        self.strategies = [
            StrategyNode(n, i, chapter)
            for i, n in enumerate(self.content)
            if n.get("type") == "strategy"
        ]

    def get_text(self) -> str:
        obj_texts = [obj.get_text() for obj in self.objectives]
        strat_texts = [strat.get_text() for strat in self.strategies]
        return "\n\n".join(obj_texts + strat_texts)

    def to_document(self) -> Dict[str, Any]:
        return {
            "text": self.get_text(),
            "metadata": {"node_type": "action_table", "id": self.attrs.get("id", "")},
        }

    def _generate_id(self) -> str:
        return f"c{self.chapter}-actions"


class DocNode(ProseMirrorNode):
    """Represents the root doc node in ProSeMirror."""

    def __init__(self, node_data: Dict[str, Any]):
        self.chapter = node_data.get("attrs", {}).get("chapter", "")
        super().__init__(node_data, 0, self.chapter)
        self.nodes = []
        self.title = ""

        heading_index = 0
        para_index = 0
        goals_index = 0
        actions_index = 0

        for i, item in enumerate(self.content):
            node_type = item.get("type", "")
            if node_type == "heading":
                heading_node = HeadingNode(item, heading_index, self.chapter)
                heading_index += 1
                # Save the first level 1 heading as the title
                if heading_node.level == 1 and not self.title:
                    self.title = heading_node.get_text()
                self.nodes.append(heading_node)
            elif node_type == "paragraph":
                para_node = ParagraphNode(item, para_index, self.chapter)
                para_index += 1
                self.nodes.append(para_node)
            elif node_type == "goals_table":
                goals_node = GoalsTableNode(item, goals_index, self.chapter)
                goals_index += 1
                self.nodes.append(goals_node)
            elif node_type == "action_table":
                actions_node = ActionTableNode(item, actions_index, self.chapter)
                actions_index += 1
                self.nodes.append(actions_node)

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
                "id": self.attrs.get("id", ""),
            },
        }

    def _generate_id(self) -> str:
        return f"chapter-{self.chapter}"

    def to_json(self) -> Dict[str, Any]:
        """Convert the doc node and all its children to a JSON-serializable dictionary."""
        result = {"type": self.type, "attrs": self.attrs, "content": []}

        for node in self.nodes:
            if isinstance(node, ProseMirrorNode):
                # If node contains direct children that are ProseMirrorNodes
                if hasattr(node, "actions") and node.actions:
                    strategy_dict = {
                        "type": node.type,
                        "attrs": node.attrs,
                        "content": [action.to_dict() for action in node.actions],
                    }
                    result["content"].append(strategy_dict)
                # If node is an ActionTableNode with objectives and strategies
                elif isinstance(node, ActionTableNode):
                    action_table_dict = {"type": node.type, "content": []}
                    # Add objectives
                    for obj in node.objectives:
                        action_table_dict["content"].append(obj.to_dict())
                    # Add strategies with their actions
                    for strat in node.strategies:
                        strat_dict = {
                            "type": strat.type,
                            "attrs": strat.attrs,
                            "content": [action.to_dict() for action in strat.actions],
                        }
                        action_table_dict["content"].append(strat_dict)

                    result["content"].append(action_table_dict)
                # For simpler nodes like paragraphs and headings
                else:
                    node_dict = {"type": node.type, "attrs": node.attrs}

                    # Handle content that contains TextNodes
                    if hasattr(node, "text_nodes") and node.text_nodes:
                        node_dict["content"] = [
                            {"type": "text", "text": text_node.text}
                            for text_node in node.text_nodes
                        ]
                    # Handle GoalsTableNode which has attrs but no content
                    elif isinstance(node, GoalsTableNode):
                        # GoalsTableNode already has its data in attrs
                        pass

                    result["content"].append(node_dict)

        return result


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


def convert_block(block, chapter, index):
    if block["type"] == "heading":
        id_suffix = block["text"].lower().replace(" ", "-")[:20]
        return {
            "type": "heading",
            "attrs": {
                "level": block["level"],
                "id": f"{chapter}-h{block['level']}-{id_suffix}",
            },
            "content": [{"type": "text", "text": block["text"]}],
        }
    elif block["type"] == "paragraph":
        return {
            "type": "paragraph",
            "attrs": {"id": f"{chapter}-p-{index}"},
            "content": [{"type": "text", "text": block["text"]}],
        }
    return None


def convert_goals(goals, chapter):
    return {
        "type": "goals_table",
        "attrs": {
            "text": goals["text"],
            "section": goals["section"],
            "values": goals["values"],
            "id": f"{chapter}-goals",
        },
    }


def convert_objective(obj, chapter, index):
    return {
        "type": "objective",
        "attrs": {
            "label": obj["label"],
            "text": obj["text"],
            "id": f"{chapter}-obj-{obj['label']}",
        },
    }


def convert_action(action, chapter, strategy_label=None):
    id_suffix = (
        f"{strategy_label}-{action['label']}" if strategy_label else action["label"]
    )
    return {
        "type": "action",
        "attrs": {
            "label": action["label"],
            "text": action["text"],
            "responsibility": action.get("responsibility", ""),
            "time_frame": action.get("time_frame", ""),
            "cost": action.get("cost", ""),
            "id": f"{chapter}-action-{id_suffix}",
        },
    }


def convert_strategy(strategy, chapter):
    strategy_result = {
        "type": "strategy",
        "attrs": {
            "label": strategy["label"],
            "text": strategy["text"],
            "id": f"{chapter}-strategy-{strategy['label']}",
        },
        "content": [],
    }

    # Convert each action with its strategy context
    for i, action in enumerate(strategy.get("actions", [])):
        strategy_result["content"].append(
            convert_action(action, chapter, strategy["label"])
        )

    return strategy_result


def convert_actions_table(actions, chapter):
    content = []

    # Add unique IDs to each objective
    for i, obj in enumerate(actions.get("objectives", [])):
        content.append(convert_objective(obj, chapter, i))

    # Add unique IDs to each strategy and its actions
    for i, strat in enumerate(actions.get("strategies", [])):
        content.append(convert_strategy(strat, chapter))

    return {
        "type": "action_table",
        "attrs": {"id": f"{chapter}-actions"},
        "content": content,
    }


def convert_document(custom_json):
    chapter = custom_json.get("chapter_number", "")
    doc = {
        "type": "doc",
        "content": [],
        "attrs": {"chapter": chapter, "id": f"chapter-{chapter}"},
    }

    # Add a chapter title first with ID
    title_text = custom_json.get("title", "")
    title_slug = title_text.lower().replace(" ", "-")[:20]
    title_block = {
        "type": "heading",
        "attrs": {"level": 1, "id": f"{chapter}-h1-{title_slug}"},
        "content": [{"type": "text", "text": title_text}],
    }

    doc["content"].append(title_block)

    # Add content blocks with IDs
    for i, item in enumerate(custom_json.get("content", [])):
        block = convert_block(item, chapter, i)
        if block:
            doc["content"].append(block)

    # Add special sections with IDs
    if "2050_goals" in custom_json:
        doc["content"].append(convert_goals(custom_json["2050_goals"], chapter))

    if "actions" in custom_json:
        doc["content"].append(convert_actions_table(custom_json["actions"], chapter))

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
