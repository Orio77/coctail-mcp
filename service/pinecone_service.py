from pinecone import Pinecone
import os
import dotenv
from models.cocktail import Cocktail
from models.ingredient import Ingredient
from config.logger import logger

dotenv.load_dotenv()

pc = Pinecone()
index = pc.Index(os.getenv('pinecone_index_name'))

def upsert_cocktail_embeddings(cocktails: list[Cocktail], embeddings: dict[int, list[float]]):
    """Upserts cocktail embeddings into Pinecone vector space."""

    # Create vectors to upsert
    vectors = _create_cocktail_vectors_to_upsert(cocktails=cocktails, embeddings=embeddings)
    logger.info(vectors)
    # Upsert vectors into Pinecone
    _upsert_vectors(vectors=vectors)

def upsert_ingredient_embeddings(ingredients: list[Ingredient], embeddings: dict[int, list[float]]):
    """Upserts ingredient embeddings into Pinecone vector space."""

    # Create vectors to upsert
    vectors = _create_ingredient_vectors_to_upsert(ingredients=ingredients, embeddings=embeddings)

    # Upsert vectors into Pinecone
    _upsert_vectors(vectors=vectors)

def _upsert_vectors(vectors: list[dict]):
    """Upserts generic vectors into Pinecone vector space."""
    response = index.upsert(vectors=vectors)
    logger.info(response)


def clear_vectorspace():
    """Clears the Pinecone vector space by deleting all vectors."""
    index.delete(delete_all=True)


def query_cocktails(top_k, vector):
    """Queries Pinecone for the most similar cocktails based on the provided vector."""
    return index.query(top_k=top_k, include_metadata=True, vector=vector)


def _create_cocktail_vectors_to_upsert(cocktails: list[Cocktail], embeddings: dict[int, list[float]]):
    """Creates a list of vectors formatted for upserting into Pinecone."""
    # Prepare vectors for upsert
    vectors: list[dict] = []

    # Create vector entries for each cocktail
    for cocktail in cocktails:
        # Build vector entry
        vectors.append({
            "id": f"cocktail_{cocktail.id}",
            "values": embeddings[cocktail.id],
            "metadata": {
                "type": "cocktail",
                "cocktail_id": cocktail.id,
                "name": cocktail.name or "",
                "tags": cocktail.tags or [],
                "instructions": cocktail.instructions or "",
                "image_url": cocktail.image_url or "",
                "ingredient_names": cocktail.ingredients or []
            }
        })

    logger.info(f"Prepared {len(vectors)} cocktail vectors for upsert.")

    return vectors

def _create_ingredient_vectors_to_upsert(ingredients: list[Ingredient], embeddings: dict[int, list[float]]):
    """Creates a list of vectors formatted for upserting ingredients into Pinecone."""
    # Prepare vectors for upsert
    vectors: list[dict] = []

    # Create vector entries for each ingredient
    for ingredient in ingredients:
        # Build vector entry
        vectors.append({
            "id": f"ingredient_{ingredient.id}",
            "values": embeddings[ingredient.id],
            "metadata": {
                "type": "ingredient",
                "ingredient_id": ingredient.id,
                "name": ingredient.name or "",
                "description": ingredient.description or "",
                "alcohol": ingredient.alcohol or False,
                "type_": ingredient.type or "",
                "image_url": ingredient.image_url or ""
            }
        })

    logger.info(f"Prepared {len(vectors)} ingredient vectors for upsert.")

    return vectors

def get_num_vectors():
    """Returns the number of vectors stored in Pinecone."""
    stats = index.describe_index_stats()
    return stats['total_vector_count']