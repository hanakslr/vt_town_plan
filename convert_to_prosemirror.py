import json
import os
import glob


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
