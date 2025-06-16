#!/usr/bin/env python3
"""
Index a ProSeMirror chapter from the Vermont Town Plan into a vector database for RAG.

This script takes ProSeMirror JSON files from output/chapters/prosemirror and indexes them
into a Qdrant vector database using LlamaIndex.

Usage:
    python index_prosemirror_chapter.py path/to/prosemirror_chapter.json
    python index_prosemirror_chapter.py --all
"""

import argparse
import json
import os
from typing import Any, List, Optional

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant.base import QdrantVectorStore
from qdrant_client import models

from convert_to_prosemirror import (
    ActionTableNode,
    DocNode,
    HeadingNode,
    create_node_from_data,
)

# Load environment variables from .env file
load_dotenv()


def process_prosemirror_node(
    node: Any,
    chapter_number: str,
    chapter_title: str,
    section_path: Optional[List[str]] = None,
) -> List[Document]:
    """
    Process a ProSeMirror node and convert it to LlamaIndex Documents.

    Args:
        node: The ProSeMirror node
        chapter_number: The chapter number for metadata
        chapter_title: The chapter title for metadata
        section_path: Optional section path for this node

    Returns:
        List of LlamaIndex Documents with appropriate metadata
    """
    documents = []

    # Base metadata for all documents
    base_metadata = {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
    }

    # Add section path if available
    if section_path:
        base_metadata["section_path"] = section_path
        # Add the last section as a specific section field
        if len(section_path) > 0:
            base_metadata["section"] = section_path[-1]

    # Get the node document with metadata
    node_doc = node.to_qdrant_document()

    # Skip empty text
    if not node_doc["text"].strip():
        return documents

    # Combine node metadata with base metadata
    combined_metadata = {**base_metadata, **node_doc["metadata"]}

    # Create document
    doc = Document(text=node_doc["text"], metadata=combined_metadata)
    documents.append(doc)

    # For certain node types, we want to process their children
    if isinstance(node, ActionTableNode):
        # Process objectives
        for objective in node.objectives:
            obj_doc = objective.to_qdrant_document()
            obj_metadata = {**base_metadata, **obj_doc["metadata"]}
            documents.append(Document(text=obj_doc["text"], metadata=obj_metadata))

        # Process strategies and their actions
        for strategy in node.strategies:
            strat_doc = strategy.to_qdrant_document()
            strat_metadata = {**base_metadata, **strat_doc["metadata"]}
            documents.append(Document(text=strat_doc["text"], metadata=strat_metadata))

            # Process actions for this strategy
            for action in strategy.actions:
                action_doc = action.to_qdrant_document()
                action_metadata = {
                    **base_metadata,
                    **action_doc["metadata"],
                    "strategy_label": strategy.label,
                }
                documents.append(
                    Document(text=action_doc["text"], metadata=action_metadata)
                )

    return documents


def extract_chapter_title(doc_node: DocNode) -> str:
    """
    Extract the chapter title from the document node.

    Args:
        doc_node: The document node

    Returns:
        The chapter title or "Unknown Chapter"
    """
    title = doc_node.get_title()
    return title if title else "Unknown Chapter"


def index_prosemirror_chapter(
    chapter_file: str, collection_name: str = "vt_town_plan_prosemirror"
) -> None:
    """
    Index a ProSeMirror chapter JSON file into Qdrant using LlamaIndex.

    Args:
        chapter_file: Path to the ProSeMirror chapter JSON file
        collection_name: Name of the Qdrant collection to use
    """
    # Load the chapter file
    with open(chapter_file, "r") as f:
        chapter_data = json.load(f)

    # Create the document node from the chapter data
    doc_node = create_node_from_data(chapter_data)
    if not isinstance(doc_node, DocNode):
        print(f"Error: {chapter_file} is not a valid ProSeMirror document")
        return

    # Extract chapter information
    chapter_number = doc_node.chapter
    chapter_title = extract_chapter_title(doc_node)

    print(f"Indexing Chapter {chapter_number}: {chapter_title}")

    # Create a list to hold all documents
    documents = []

    # Build section path as we traverse the document
    current_section_path = []
    last_heading_level = 0

    # Process nodes
    for node in doc_node.nodes:
        # Track section path based on heading levels
        if isinstance(node, HeadingNode):
            # Update section path based on heading level
            if node.level <= last_heading_level:
                # Remove elements from path to match the new level
                current_section_path = current_section_path[: node.level - 1]

            # Add the current heading to the path
            if node.level <= len(current_section_path) + 1:
                if node.level == len(current_section_path) + 1:
                    current_section_path.append(node.get_text())
                else:
                    current_section_path[node.level - 1] = node.get_text()

            last_heading_level = node.level

        # Process the node
        node_docs = process_prosemirror_node(
            node, chapter_number, chapter_title, current_section_path.copy()
        )
        documents.extend(node_docs)

    # Print summary of documents
    print(f"Prepared {len(documents)} documents for indexing")

    # Set up the vector database connection
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_CLOUD_API_KEY", None),
    )

    # Configure LlamaIndex
    Settings.embed_model = OpenAIEmbedding()
    Settings.node_parser = SimpleNodeParser.from_defaults()

    # Check if collection exists before creating it
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if collection_name not in collection_names:
        print(f"Creating new collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )
    else:
        print(f"Using existing collection: {collection_name}")

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
        print(f"  Type: {metadata.get('node_type', 'N/A')}")
        if "section" in metadata:
            print(f"  Section: {metadata.get('section')}")


def index_all_prosemirror_chapters(
    directory: str = "output/chapters/prosemirror",
    collection_name: str = "vt_town_plan_prosemirror",
) -> None:
    """
    Index all ProSeMirror JSON files in the specified directory.

    Args:
        directory: Path to the directory containing ProSeMirror chapter JSON files
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
        index_prosemirror_chapter(chapter_file, collection_name)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index ProSeMirror chapter files into a vector database for RAG."
    )

    # Create a mutually exclusive group for file input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "chapter_file",
        nargs="?",  # Make positional argument optional
        help="Path to the ProSeMirror chapter JSON file to index",
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="Index all ProSeMirror JSON files in the output/chapters/prosemirror directory",
    )

    parser.add_argument(
        "--collection",
        default="vt_town_plan_prosemirror",
        help="Name of the Qdrant collection to use",
    )

    args = parser.parse_args()

    if args.all:
        # Index all files in output/chapters/prosemirror
        index_all_prosemirror_chapters("output/chapters/prosemirror", args.collection)
    else:
        # Index a single file
        index_prosemirror_chapter(args.chapter_file, args.collection)


if __name__ == "__main__":
    main()
