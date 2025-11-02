from typing import Dict, List
import ollama
import os
import dotenv
from models.cocktail import Cocktail
from models.ingredient import Ingredient
from config.logger import logger

dotenv.load_dotenv()

# Constants
EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL"

def _get_embedding_model() -> str:
    """Get embedding model from environment variable."""
    model = os.getenv(EMBEDDING_MODEL_ENV_VAR)
    if not model:
        raise ValueError(f"{EMBEDDING_MODEL_ENV_VAR} environment variable is not set")
    return model

def _create_embedding(model: str, input_text: str) -> List[float]:
    """Create embedding for given input text."""
    try:
        embedding_response = ollama.embed(model=model, input=input_text)
        if not embedding_response.embeddings:
            raise ValueError("No embedding returned from model")
        return [float(val) for val in embedding_response.embeddings[0]]
    except Exception as e:
        logger.error(f"Failed to create embedding for input '{input_text[:50]}...': {e}")
        raise

def create_cocktail_embeddings(cocktails: List[Cocktail]) -> Dict[str, List[float]]:
    """Creates and returns a mapping of cocktail IDs to their embeddings."""
    if not cocktails:
        raise ValueError("Cocktails list cannot be empty")
    
    logger.info("Creating cocktail embeddings...")
    model = _get_embedding_model()
    logger.info(f"Using embedding model: {model}")

    embeddings_map = {}

    for cocktail in cocktails:
        try:
            embedding = _create_embedding(model, repr(cocktail))
            embeddings_map[cocktail.id] = embedding
        except Exception as e:
            logger.error(f"Failed to create embedding for cocktail {cocktail.id}: {e}")
            continue

    if not embeddings_map:
        raise RuntimeError("Failed to create any cocktail embeddings")

    logger.info(f"Created {len(embeddings_map)} cocktail embeddings.")
    return embeddings_map

def create_ingredient_embeddings(ingredients: List[Ingredient]) -> Dict[str, List[float]]:
    """Creates and returns a mapping of ingredient IDs to their embeddings."""
    if not ingredients:
        raise ValueError("Ingredients list cannot be empty")
    
    logger.info("Creating ingredient embeddings...")
    model = _get_embedding_model()
    logger.info(f"Using embedding model: {model}")

    embeddings_map = {}

    for ingredient in ingredients:
        try:
            embedding = _create_embedding(model, repr(ingredient))
            embeddings_map[ingredient.id] = embedding
        except Exception as e:
            logger.error(f"Failed to create embedding for ingredient {ingredient.id}: {e}")
            continue

    if not embeddings_map:
        raise RuntimeError("Failed to create any ingredient embeddings")

    logger.info(f"Created {len(embeddings_map)} ingredient embeddings.")
    return embeddings_map

def embed_query(query: str) -> List[float]:
    """Creates and returns the embedding for a given query string."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace")
    
    model = _get_embedding_model()
    logger.info(f"Embedding query using model: {model}")

    try:
        return _create_embedding(model, query.strip())
    except Exception as e:
        logger.error(f"Failed to embed query '{query}': {e}")
        raise