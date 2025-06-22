#!/usr/bin/env python3
"""
Index a chapter from the Vermont Town Plan into a vector database for RAG.

This script takes a JSON file containing a structured chapter document and indexes it into
a Qdrant vector database using LlamaIndex.

Usage:
    python index_chapter.py path/to/chapter.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant.base import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()


def process_content_item(
    item: Dict[str, Any],
    chapter_number: str,
    chapter_title: str,
    section_path: Optional[List[str]] = None,
) -> Document:
    """
    Process a content item from the chapter document and convert it to a LlamaIndex Document.

    Args:
        item: The content item from the chapter JSON
        chapter_number: The chapter number for metadata
        chapter_title: The chapter title for metadata
        section_path: Optional section path for this item

    Returns:
        LlamaIndex Document with appropriate metadata
    """
    # Extract the text based on item type
    text = item.get("text", "")

    # Prepare metadata
    metadata = {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "item_type": item.get("type", "unknown"),
    }

    # Add section path if available
    if section_path or item.get("section_path"):
        path = section_path or item.get("section_path", [])
        metadata["section_path"] = path

        # Add the last section as a specific section field for easier querying
        if path and len(path) > 0:
            metadata["section"] = path[-1]

    # Add additional metadata based on item type
    if item.get("type") == "heading":
        metadata["heading_level"] = item.get("level", 1)
    elif item.get("type") == "paragraph":
        metadata["paragraph_style"] = item.get("paragraph_style", "")

    # Create and return the document
    return Document(text=text, metadata=metadata)


def process_special_section(
    section_data: Dict[str, Any],
    section_name: str,
    chapter_number: str,
    chapter_title: str,
) -> List[Document]:
    """
    Process a special section like 2050_goals, three_facts, etc.

    Args:
        section_data: The section data from the chapter JSON
        section_name: The name of the section (e.g., "2050_goals", "three_facts")
        chapter_number: The chapter number for metadata
        chapter_title: The chapter title for metadata

    Returns:
        List of LlamaIndex Documents representing this section
    """
    documents = []

    # Extract text and section info
    text = section_data.get("text", "")
    section = section_data.get("section", "")

    # Base metadata for all documents from this section
    base_metadata = {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "item_type": section_name,
        "section": section,
    }

    # Process main section text if not empty
    if text:
        doc = Document(text=text, metadata={**base_metadata, "content_type": "header"})
        documents.append(doc)

    # Process specific section types
    if section_name == "2050_goals" and "values" in section_data:
        # Process 2050 goals values
        goals_text = "\n".join(
            [f"{key}: {value}" for key, value in section_data["values"].items()]
        )
        doc = Document(
            text=goals_text, metadata={**base_metadata, "content_type": "goals"}
        )
        documents.append(doc)

    elif (
        section_name in ["three_facts", "public_engagement"] and "facts" in section_data
    ):
        # Process facts as individual documents
        for i, fact in enumerate(section_data["facts"]):
            fact_text = f"{fact.get('title', '')}\n{fact.get('text', '')}"
            doc = Document(
                text=fact_text,
                metadata={**base_metadata, "content_type": "fact", "fact_index": i},
            )
            documents.append(doc)

    elif section_name == "actions":
        # Process objectives
        if "objectives" in section_data:
            for obj in section_data["objectives"]:
                obj_text = f"{obj.get('label', '')}: {obj.get('text', '')}"
                doc = Document(
                    text=obj_text,
                    metadata={
                        **base_metadata,
                        "content_type": "objective",
                        "label": obj.get("label", ""),
                    },
                )
                documents.append(doc)

        # Process strategies and actions
        if "strategies" in section_data:
            for strategy in section_data["strategies"]:
                # Add the strategy itself
                strat_text = f"{strategy.get('label', '')}: {strategy.get('text', '')}"
                doc = Document(
                    text=strat_text,
                    metadata={
                        **base_metadata,
                        "content_type": "strategy",
                        "label": strategy.get("label", ""),
                    },
                )
                documents.append(doc)

                # Add each action under this strategy
                if "actions" in strategy:
                    for action in strategy["actions"]:
                        action_text = (
                            f"{action.get('label', '')}: {action.get('text', '')}\n"
                            f"Responsibility: {action.get('responsibility', '')}\n"
                            f"Time Frame: {action.get('time_frame', '')}\n"
                            f"Cost: {action.get('cost', '')}"
                        )

                        action_metadata = {
                            **base_metadata,
                            "content_type": "action",
                            "label": action.get("label", ""),
                            "strategy_label": strategy.get("label", ""),
                            "responsibility": action.get("responsibility", ""),
                            "time_frame": action.get("time_frame", ""),
                            "cost": action.get("cost", ""),
                        }

                        # Add starred and multiple_strategies flags if present
                        if action.get("starred"):
                            action_metadata["starred"] = True
                        if action.get("multiple_strategies"):
                            action_metadata["multiple_strategies"] = True

                        doc = Document(text=action_text, metadata=action_metadata)
                        documents.append(doc)

    return documents


def index_chapter(chapter_file: str, collection_name: str = "vt_town_plan") -> None:
    """
    Index a chapter JSON file into Qdrant using LlamaIndex.

    Args:
        chapter_file: Path to the chapter JSON file
        collection_name: Name of the Qdrant collection to use
    """
    # Load the chapter file
    with open(chapter_file, "r") as f:
        chapter_data = json.load(f)

    # Extract basic chapter information
    chapter_number = chapter_data.get("chapter_number", "unknown")
    chapter_title = chapter_data.get("title", "unknown")

    print(f"Indexing Chapter {chapter_number}: {chapter_title}")

    # Create a list to hold all documents
    documents = []

    # Process content items
    content_items = chapter_data.get("content", [])
    for item in content_items:
        doc = process_content_item(item, chapter_number, chapter_title)
        documents.append(doc)

    # Process special sections
    special_sections = {
        "2050_goals": chapter_data.get("2050_goals"),
        "three_facts": chapter_data.get("three_facts"),
        "public_engagement": chapter_data.get("public_engagement"),
        "actions": chapter_data.get("actions"),
    }

    for section_name, section_data in special_sections.items():
        if section_data:
            section_docs = process_special_section(
                section_data, section_name, chapter_number, chapter_title
            )
            documents.extend(section_docs)

    # Print summary of documents
    print(f"Prepared {len(documents)} documents for indexing")

    # Set up the vector database connection
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", None),
    )

    # Configure LlamaIndex
    Settings.embed_model = OpenAIEmbedding()
    Settings.node_parser = SimpleNodeParser.from_defaults()

    # Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create and save the index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    print(
        f"Indexed {len(documents)} documents to Qdrant collection '{collection_name}'"
    )

    # Provide an example query using the index
    example_query = f"What are the main objectives in Chapter {chapter_number}?"
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(example_query)

    print("\nExample query retrieval test:")
    print(f"Query: {example_query}")
    print(f"Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        metadata = node.metadata
        print(f"\nNode {i + 1}:")
        print(f"  Text: {node.text[:100]}...")
        print(f"  Score: {node.score}")
        print(f"  Chapter: {metadata.get('chapter_number')}")
        print(
            f"  Type: {metadata.get('item_type')} - {metadata.get('content_type', 'N/A')}"
        )
        if "section" in metadata:
            print(f"  Section: {metadata.get('section')}")


def index_all_chapters(directory: str, collection_name: str = "vt_town_plan") -> None:
    """
    Index all JSON files in the specified directory.

    Args:
        directory: Path to the directory containing chapter JSON files
        collection_name: Name of the Qdrant collection to use
    """
    import glob
    import os

    # Find all JSON files in the directory
    chapter_files = glob.glob(os.path.join(directory, "*.json"))

    if not chapter_files:
        print(f"No JSON files found in {directory}")
        return

    print(f"Found {len(chapter_files)} chapter files to index")

    # Index each file
    for i, chapter_file in enumerate(sorted(chapter_files)):
        print(
            f"\n[{i + 1}/{len(chapter_files)}] Processing: {os.path.basename(chapter_file)}"
        )
        index_chapter(chapter_file, collection_name)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index a chapter JSON file into a vector database for RAG."
    )

    # Create a mutually exclusive group for file input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "chapter_file",
        nargs="?",  # Make positional argument optional
        help="Path to the chapter JSON file to index",
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Index all JSON files in the output/chapters directory",
    )

    parser.add_argument(
        "--collection",
        default="vt_town_plan",
        help="Name of the Qdrant collection to use",
    )

    args = parser.parse_args()

    if args.all:
        # Index all files in output/chapters
        index_all_chapters("output/chapters", args.collection)
    else:
        # Index a single file
        index_chapter(args.chapter_file, args.collection)


if __name__ == "__main__":
    main()
