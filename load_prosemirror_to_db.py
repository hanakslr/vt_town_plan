#!/usr/bin/env python3
"""
Load ProSeMirror JSON files into PostgreSQL database.

This script reads ProSeMirror JSON files from output/chapters/prosemirror and loads them
into a PostgreSQL database. The database connection URL should be set in the DATABASE_URL
environment variable.

Usage:
    python load_prosemirror_to_db.py
"""

import glob
import json
import os
from typing import Any, Dict

import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_db_connection():
    """Create a database connection using SUPABASE_DB_URL environment variable."""
    # Fetch variables
    USER = os.getenv("SUPABASE_USER")
    PASSWORD = os.getenv("SUPABASE_PASSWORD")
    HOST = os.getenv("SUPABASE_HOST")
    PORT = os.getenv("SUPABASE_PORT")
    DBNAME = os.getenv("SUPABASE_DATABASE")
    URL = os.getenv("SUPABASE_CONNECTION_URL")

    print(USER, PASSWORD, HOST, PORT, DBNAME)

    # Connect to the database
    connection = psycopg2.connect(URL)

    print("Connection sucessful")

    return connection


def extract_chapter_info(doc: Dict[str, Any]) -> tuple[str, str, str]:
    """
    Extract chapter number, title, and version from the document.

    Args:
        doc: The ProSeMirror document

    Returns:
        Tuple of (chapter_number, title)
    """
    # Get chapter number from attrs
    chapter_number = doc.get("attrs", {}).get("chapter", "")

    # Find the title in the first heading node
    title = ""

    for node in doc.get("content", []):
        if node.get("type") == "heading" and node.get("attrs", {}).get("level") == 1:
            # Get text from the first text node in the heading
            for content_node in node.get("content", []):
                if content_node.get("type") == "text":
                    title = content_node.get("text", "")
                    break
            break

    return chapter_number.strip(), title.strip()


def load_file_to_db(file_path: str, cur) -> bool:
    """
    Load a single ProSeMirror file into the database.

    Args:
        file_path: Path to the ProSeMirror JSON file
        cur: Database cursor

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        # Extract chapter information
        chapter_number, title = extract_chapter_info(doc)

        # Insert into database
        cur.execute(
            """
            INSERT INTO chapters (collection, title, version, content)
            VALUES (%s, %s, %s, %s)
            """,
            ("chapters", title, 1, json.dumps(doc)),
        )

        print(f"Loaded chapter {chapter_number}: {title}")
        return True

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False


def main():
    """Main entry point."""
    # Get all JSON files in the prosemirror directory
    input_dir = "output/chapters/prosemirror"
    input_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not input_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(input_files)} files to process")

    try:
        # Connect to database
        conn = get_db_connection()
        cur = conn.cursor()

        # Process each file
        successful = 0
        for input_file in sorted(input_files):
            if load_file_to_db(input_file, cur):
                successful += 1

        # Commit changes
        conn.commit()

        print(
            f"\nLoad complete: {successful}/{len(input_files)} files loaded successfully"
        )

    except Exception as e:
        print(f"Error: {e}")
        if "conn" in locals():
            conn.rollback()
    finally:
        if "cur" in locals():
            cur.close()
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
