#!/usr/bin/env python3
"""
Query the Vermont Town Plan vector database for information.

This script connects to a Qdrant vector database containing indexed Vermont Town Plan
chapters and allows querying for information.

Usage:
    python query_index.py "What are the objectives for transportation?"
"""

import argparse
import os
import sys
from typing import List

import qdrant_client
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant.base import QdrantVectorStore

# Load environment variables from .env file
load_dotenv()


def format_source_nodes(source_nodes: List[NodeWithScore]) -> str:
    """
    Format the source nodes into a readable string with citation information.

    Args:
        source_nodes: List of retrieved nodes with their relevance scores

    Returns:
        Formatted string with source information
    """
    if not source_nodes:
        return "No sources found."

    sources_text = []
    for i, node in enumerate(source_nodes):
        metadata = node.metadata

        # Extract source information
        chapter_number = metadata.get("chapter_number", "Unknown")
        chapter_title = metadata.get("chapter_title", "Unknown")
        content_type = metadata.get("content_type", "")
        item_type = metadata.get("item_type", "")
        section = metadata.get("section", "")

        # Format specific source information based on content type
        source_detail = ""
        if item_type == "actions":
            if content_type == "objective":
                source_detail = f"Objective {metadata.get('label', '')}"
            elif content_type == "strategy":
                source_detail = f"Strategy {metadata.get('label', '')}"
            elif content_type == "action":
                source_detail = f"Action {metadata.get('label', '')} (Strategy {metadata.get('strategy_label', '')})"

        # Build the source citation
        source_citation = f"Chapter {chapter_number}: {chapter_title}"
        if section:
            source_citation += f" - Section: {section}"
        if source_detail:
            source_citation += f" - {source_detail}"

        # Include the text snippet and relevance score
        text_snippet = node.text[:200] + "..." if len(node.text) > 200 else node.text
        source_info = f"[{i + 1}] {source_citation}\nRelevance: {node.score:.2f}\nExcerpt: {text_snippet}\n"
        sources_text.append(source_info)

    return "\n".join(sources_text)


def create_query_engine(
    collection_name: str = "vt_town_plan", top_k: int = 5
) -> RetrieverQueryEngine:
    """
    Create a query engine connected to the Qdrant vector database.

    Args:
        collection_name: Name of the Qdrant collection to query
        top_k: Number of top matches to retrieve

    Returns:
        A query engine that can be used to ask questions
    """
    # Make sure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Configure LlamaIndex
    Settings.embed_model = OpenAIEmbedding()

    # Set up the vector database connection
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY", None),
    )

    # Set up Qdrant vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )

    # Create the index from the existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store)

    # Create a retriever with the specified top_k
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # Add a similarity cutoff to filter out low-relevance results
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.7)]

    # Create and return the query engine
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=node_postprocessors,
    )


def query_plan(
    query_text: str,
    collection_name: str = "vt_town_plan",
    top_k: int = 5,
    show_sources: bool = True,
) -> str:
    """
    Query the Vermont Town Plan and get a response.

    Args:
        query_text: The question to ask
        collection_name: Name of the Qdrant collection to query
        top_k: Number of top matches to retrieve
        show_sources: Whether to include source information in the response

    Returns:
        A string containing the answer and optionally source information
    """
    print(f"Query: {query_text}")

    try:
        # Create the query engine
        query_engine = create_query_engine(collection_name, top_k)

        # Execute the query
        response = query_engine.query(query_text)

        # Format the response
        if show_sources and hasattr(response, "source_nodes"):
            sources = format_source_nodes(response.source_nodes)
            return f"{response}\n\nSources:\n{sources}"
        else:
            return str(response)

    except Exception as e:
        return f"Error querying the index: {str(e)}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query the Vermont Town Plan vector database."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="The question to ask the Vermont Town Plan",
    )
    parser.add_argument(
        "--collection",
        default="vt_town_plan",
        help="Name of the Qdrant collection to query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to retrieve",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source information in the response",
    )

    args = parser.parse_args()

    # Combine all query arguments into a single string
    query_text = " ".join(args.query)

    # Query the index
    response = query_plan(
        query_text=query_text,
        collection_name=args.collection,
        top_k=args.top_k,
        show_sources=not args.no_sources,
    )

    # Print the response
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
