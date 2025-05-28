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
docker-compose down -v
```

## Running

1. Extract structure of the docx file. `extract.py file_name`. This will dump structured JSON to ouput/
2. Transform the data. Using sentence transformers to generated embeddings in `transform.py`
3. Load the data into
