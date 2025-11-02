# Cocktail Recommendation System

A Python-based cocktail recommendation system that uses RAG (Retrieval-Augmented Generation) with vector embeddings and Pinecone for intelligent cocktail suggestions. The system integrates with LMStudio for local language model inference.

## Features

- 🍹 Cocktail data management and search
- 🔍 Vector-based similarity search using embeddings
- 🌲 Pinecone vector database integration
- 🤖 RAG (Retrieval-Augmented Generation) for intelligent recommendations
- 🖥️ LMStudio integration for local LLM inference
- 📊 Comprehensive logging and error handling
- 🧪 Full test coverage

## Project Structure

```
├── cocktail_server.py          # Main server application
├── config/
│   └── logger.py              # Logging configuration
├── data/
│   └── cocktail_dataset.json  # Cocktail database
├── models/
│   ├── cocktail.py           # Cocktail data model
│   └── ingredient.py         # Ingredient data model
├── services/
│   ├── cocktail_data_service.py  # Cocktail data operations
│   ├── embedding_service.py      # Vector embedding generation
│   ├── pinecone_service.py       # Pinecone vector DB operations
│   └── rag_service.py            # RAG implementation
├── tests/                     # Unit tests
└── logs/                      # Application logs
```

## Prerequisites

- Python 3.13 (but previous versions may work)
- LMStudio installed and running
- Pinecone account, API key and Index created

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create an Anaconda environment:**

   ```bash
   conda create -n cocktail-env python=3.13 -y
   conda activate cocktail-env
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` file with your configuration:

   ```
   data_path=your/path/to/cocktail_dataset.json
   embedding_model=your_embedding_model_here
   embedding_dimension=your_embedding_dimension_here
   pinecone_index_name=your_pinecone_index_name_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## LMStudio Integration Setup

1. **Install LMStudio:**

   - Download from [https://lmstudio.ai/](https://lmstudio.ai/)
   - Install and launch the application

2. **Download a Model:**

   - Open LMStudio
   - Browse and download a suitable model
   - Recommended model for cocktail recommendations:
     - `qwen3-8b`

3. **Integrate the MCP**

   - In LMStudio, go to Settings -> Program -> Install -> Edit mcp.json
   - Add the following json there:

   ```json
   {
   	"mcpServers": {
   		"cocktail-mcp": {
   			"command": "C:/path/to/your/environment/python.exe", <!-- mine is "C:/Users/macie/anaconda3/envs/cocktail-mcp/python.exe" -->
   			"args": ["c:\\path\\to\\cocktail_server.py"]
   		}
   	}
   }
   ```

## _Usage_

- Simply chat in the LMStudio interface and ask for cocktail recommendations!

## Running Tests

Execute the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rag_service.py
```

### Logs

Check application logs in the `logs/` directory:

```bash
tail -f logs/cocktail_app.log
```
