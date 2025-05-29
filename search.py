"""
Command line semantic search of what is in the local db.
"""
import psycopg
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform semantic search against town plan"
    )

    parser.add_argument("search", help="Search term")

    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good for retrieval
    embedding = model.encode(args.search, convert_to_numpy=True)
    embedding = embedding.tolist()

    conn = psycopg.connect(
        dbname="db",
        user="admin",
        password="password",
        host="localhost",
        port=5431
    )


    cur = conn.cursor()

    cur.execute(
        """
        SELECT content_type, content, section_path
        FROM plan_chunks
        WHERE content_type = 'paragraph'
        ORDER BY embedding <-> %s::vector
        LIMIT 5;
        """,
        (embedding,))
    
    results = cur.fetchall()
    print("\nTop 5 most similar chunks:")
    print("-" * 80)
    for i, (content_type, content, section_path) in enumerate(results, 1):
        print(f"\n{i}. Type: {content_type}")
        if section_path:
            print(f"   Section: {' > '.join(section_path)}")
        print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
        print("-" * 80)

    cur.close()
    conn.close()