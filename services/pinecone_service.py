from pinecone import Pinecone
import os
import dotenv
from typing import List, Dict, Optional, Any
from models.cocktail import Cocktail
from models.ingredient import Ingredient
from config.logger import logger

dotenv.load_dotenv()

# Initialize Pinecone with error handling
try:
    pc = Pinecone()
    index_name = os.getenv('PINECONE_INDEX_NAME')
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME environment variable not set")
    index = pc.Index(index_name)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise


class PineconeError(Exception):
    """Custom exception for Pinecone-related errors."""
    pass


def upsert_cocktail_embeddings(cocktails: List[Cocktail], embeddings: Dict[int, List[float]]) -> None:
    """Upserts cocktail embeddings into Pinecone vector space."""
    if not cocktails:
        logger.warning("No cocktails provided for upsert")
        return
    
    if not embeddings:
        logger.warning("No embeddings provided for upsert")
        return

    try:
        # Create vectors to upsert
        vectors = _create_cocktail_vectors_to_upsert(cocktails=cocktails, embeddings=embeddings)
        
        if not vectors:
            logger.warning("No valid vectors created for cocktails")
            return
            
        logger.info(f"Upserting {len(vectors)} cocktail vectors")
        
        # Upsert vectors into Pinecone
        _upsert_vectors(vectors=vectors)
        
    except Exception as e:
        logger.error(f"Failed to upsert cocktail embeddings: {e}")
        raise PineconeError(f"Cocktail embedding upsert failed: {e}") from e


def upsert_ingredient_embeddings(ingredients: List[Ingredient], embeddings: Dict[int, List[float]]) -> None:
    """Upserts ingredient embeddings into Pinecone vector space."""
    if not ingredients:
        logger.warning("No ingredients provided for upsert")
        return
    
    if not embeddings:
        logger.warning("No embeddings provided for upsert")
        return

    try:
        # Create vectors to upsert
        vectors = _create_ingredient_vectors_to_upsert(ingredients=ingredients, embeddings=embeddings)
        
        if not vectors:
            logger.warning("No valid vectors created for ingredients")
            return
            
        logger.info(f"Upserting {len(vectors)} ingredient vectors")
        
        # Upsert vectors into Pinecone
        _upsert_vectors(vectors=vectors)
        
    except Exception as e:
        logger.error(f"Failed to upsert ingredient embeddings: {e}")
        raise PineconeError(f"Ingredient embedding upsert failed: {e}") from e


def _upsert_vectors(vectors: List[Dict[str, Any]]) -> None:
    """Upserts generic vectors into Pinecone vector space."""
    if not vectors:
        logger.warning("No vectors to upsert")
        return
    
    # Batch size limit for Pinecone (usually 100)
    batch_size = 100
    
    try:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            response = index.upsert(vectors=batch)
            logger.debug(f"Upserted batch {i//batch_size + 1}: {response}")
            
        logger.info(f"Successfully upserted {len(vectors)} vectors")
        
    except Exception as e:
        logger.error(f"Failed to upsert vectors: {e}")
        raise PineconeError(f"Vector upsert failed: {e}") from e


def clear_vectorspace() -> None:
    """Clears the Pinecone vector space by deleting all vectors."""
    try:
        response = index.delete(delete_all=True)
        logger.info(f"Cleared vector space: {response}")
    except Exception as e:
        logger.error(f"Failed to clear vector space: {e}")
        raise PineconeError(f"Vector space clear failed: {e}") from e


def query_cocktails(top_k: int, vector: List[float]) -> Optional[Any]:
    """Queries Pinecone for the most similar cocktails based on the provided vector."""
    if not vector:
        logger.warning("Empty vector provided for query")
        return None
    
    if top_k <= 0:
        logger.warning(f"Invalid top_k value: {top_k}. Using default of 5")
        top_k = 5
    
    # Limit top_k to reasonable maximum
    max_top_k = 1000
    if top_k > max_top_k:
        logger.warning(f"top_k {top_k} exceeds maximum {max_top_k}. Using {max_top_k}")
        top_k = max_top_k
    
    try:
        result = index.query(top_k=top_k, include_metadata=True, vector=vector)
        logger.debug(f"Query returned {len(result.matches) if hasattr(result, 'matches') and result.matches else 0} matches")
        return result
        
    except Exception as e:
        logger.error(f"Failed to query cocktails: {e}")
        raise PineconeError(f"Cocktail query failed: {e}") from e


def _create_cocktail_vectors_to_upsert(cocktails: List[Cocktail], embeddings: Dict[int, List[float]]) -> List[Dict[str, Any]]:
    """Creates a list of vectors formatted for upserting into Pinecone."""
    vectors: List[Dict[str, Any]] = []
    skipped_count = 0

    for cocktail in cocktails:
        try:
            # Validate cocktail
            if not _is_valid_cocktail(cocktail):
                logger.warning(f"Skipping invalid cocktail: {cocktail}")
                skipped_count += 1
                continue
            
            # Check if embedding exists
            if cocktail.id not in embeddings:
                logger.warning(f"No embedding found for cocktail ID {cocktail.id}")
                skipped_count += 1
                continue
            
            embedding = embeddings[cocktail.id]
            if not embedding or not isinstance(embedding, list):
                logger.warning(f"Invalid embedding for cocktail ID {cocktail.id}")
                skipped_count += 1
                continue
            
            # Build vector entry with safe defaults
            vector_entry = {
                "id": f"cocktail_{cocktail.id}",
                "values": embedding,
                "metadata": {
                    "type": "cocktail",
                    "cocktail_id": cocktail.id,
                    "name": _safe_string(cocktail.name),
                    "tags": _safe_list(cocktail.tags),
                    "instructions": _safe_string(cocktail.instructions),
                    "image_url": _safe_string(cocktail.image_url),
                    "ingredient_names": _safe_list(cocktail.ingredients)
                }
            }
            
            vectors.append(vector_entry)
            
        except Exception as e:
            logger.warning(f"Error processing cocktail {getattr(cocktail, 'id', 'unknown')}: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} invalid cocktails")
    
    logger.info(f"Prepared {len(vectors)} cocktail vectors for upsert")
    return vectors


def _create_ingredient_vectors_to_upsert(ingredients: List[Ingredient], embeddings: Dict[int, List[float]]) -> List[Dict[str, Any]]:
    """Creates a list of vectors formatted for upserting ingredients into Pinecone."""
    vectors: List[Dict[str, Any]] = []
    skipped_count = 0

    for ingredient in ingredients:
        try:
            # Validate ingredient
            if not _is_valid_ingredient(ingredient):
                logger.warning(f"Skipping invalid ingredient: {ingredient}")
                skipped_count += 1
                continue
            
            # Check if embedding exists
            if ingredient.id not in embeddings:
                logger.warning(f"No embedding found for ingredient ID {ingredient.id}")
                skipped_count += 1
                continue
            
            embedding = embeddings[ingredient.id]
            if not embedding or not isinstance(embedding, list):
                logger.warning(f"Invalid embedding for ingredient ID {ingredient.id}")
                skipped_count += 1
                continue
            
            # Build vector entry with safe defaults
            vector_entry = {
                "id": f"ingredient_{ingredient.id}",
                "values": embedding,
                "metadata": {
                    "type": "ingredient",
                    "ingredient_id": ingredient.id,
                    "name": _safe_string(ingredient.name),
                    "description": _safe_string(ingredient.description),
                    "alcohol": _safe_bool(ingredient.alcohol),
                    "type_": _safe_string(ingredient.type),
                    "image_url": _safe_string(ingredient.image_url)
                }
            }
            
            vectors.append(vector_entry)
            
        except Exception as e:
            logger.warning(f"Error processing ingredient {getattr(ingredient, 'id', 'unknown')}: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} invalid ingredients")
    
    logger.info(f"Prepared {len(vectors)} ingredient vectors for upsert")
    return vectors


def get_num_vectors() -> int:
    """Returns the number of vectors stored in Pinecone."""
    try:
        stats = index.describe_index_stats()
        total_count = stats.get('total_vector_count', 0)
        logger.debug(f"Vector count: {total_count}")
        return total_count
        
    except Exception as e:
        logger.error(f"Failed to get vector count: {e}")
        raise PineconeError(f"Failed to get vector count: {e}") from e


# Helper functions for validation and safe value extraction
def _is_valid_cocktail(cocktail: Cocktail) -> bool:
    """Validate cocktail object."""
    return (
        cocktail is not None and 
        hasattr(cocktail, 'id') and 
        cocktail.id is not None and
        isinstance(cocktail.id, (int, str))
    )


def _is_valid_ingredient(ingredient: Ingredient) -> bool:
    """Validate ingredient object."""
    return (
        ingredient is not None and 
        hasattr(ingredient, 'id') and 
        ingredient.id is not None and
        isinstance(ingredient.id, (int, str))
    )


def _safe_string(value: Any) -> str:
    """Safely convert value to string."""
    if value is None:
        return ""
    return str(value)


def _safe_list(value: Any) -> List[Any]:
    """Safely convert value to list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _safe_bool(value: Any) -> bool:
    """Safely convert value to boolean."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)