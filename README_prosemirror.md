# ProSeMirror Indexing for Vermont Town Plan

This extension to the Vermont Town Plan project provides functionality to index ProSeMirror JSON documents into a vector database for RAG (Retrieval Augmented Generation).

## Setup

1. Make sure you have all required dependencies:
```bash
pip install qdrant-client llama-index-embeddings-openai llama-index-vector-stores-qdrant python-dotenv
```

2. Ensure that your `.env` file contains the necessary API keys:
```
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=http://localhost:6333  # Adjust as needed
QDRANT_API_KEY=your-qdrant-api-key  # If required
```

## Usage

### Generate ProSeMirror Files

1. First, generate the ProSeMirror files from the chapter JSON files:
```bash
python convert_to_prosemirror.py
```

This will create ProSeMirror JSON files in the `output/chapters/prosemirror` directory.

### Index ProSeMirror Files

You can index ProSeMirror files in two ways:

1. Index a single ProSeMirror file:
```bash
python index_prosemirror_chapter.py /path/to/prosemirror_file.json
```

2. Index all ProSeMirror files in the default directory:
```bash
python index_prosemirror_chapter.py --all
```

3. Specify a custom collection name:
```bash
python index_prosemirror_chapter.py --all --collection my_custom_collection
```

## Structure

The system uses a class hierarchy to represent ProSeMirror nodes:

- `ProseMirrorNode`: Base class for all node types
- Specific node classes for different content types:
  - `DocNode`: The root document node
  - `HeadingNode`: For headings
  - `ParagraphNode`: For paragraphs
  - `GoalsTableNode`: For 2050 goals tables
  - `ActionTableNode`: For action tables
  - `ObjectiveNode`: For objectives
  - `StrategyNode`: For strategies
  - `ActionNode`: For actions

Each node type knows how to extract its text content and create appropriate metadata for indexing.