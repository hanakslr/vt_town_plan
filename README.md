# Vermont Town Plan Helper

Extract structure from a Vermont Town Plan, generate embeddings to perform semantic search.

## Python env setup

```
# Create a virtual environment
uv venv

# Install dependencies
uv sync

# Activate virtual env - this will happen automatically after the first time
source .venv/bin/activate
```

## Database Setup

This project uses pgvector to store the embeddings. It only runs locally right now.

```
# Up the db
docker-compose -f docker/db.yml up -d

# Down the db (to reset it)
docker compose -f docker/db.yml down -v
```

## Running

1. Extract structure and embeddings of the docx file. `python extract.py file_name`. This will read, chunk, embed, and dump to pg.
2. Search. Running `python search.py search_term` will perform a semantic search and return the top 5 results.

## Qdrant

docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

docker stop qdrant && docker rm qdrant

http://localhost:6333/dashboard
