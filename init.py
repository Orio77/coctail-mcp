from config.logger import logger

from services.cocktail_data_service import create_cocktails, create_ingredients
from services.pinecone_service import get_num_vectors, clear_vectorspace, upsert_cocktail_embeddings, upsert_ingredient_embeddings
from services.embedding_service import create_cocktail_embeddings, create_ingredient_embeddings


cocktails = create_cocktails()
ingredients = create_ingredients()
vector_count = get_num_vectors()

if len(cocktails) + len(ingredients) != vector_count:
    logger.info(f"Vector count ({vector_count}) does not match cocktail count ({len(cocktails)}) and ingredient count ({len(ingredients)}). Rebuilding vector space...")
    if vector_count > 0:
        clear_vectorspace()
    cocktail_embeddings = create_cocktail_embeddings(cocktails=cocktails)
    ingredient_embeddings = create_ingredient_embeddings(ingredients=ingredients)
    upsert_cocktail_embeddings(cocktails=cocktails, embeddings=cocktail_embeddings)
    upsert_ingredient_embeddings(ingredients=ingredients, embeddings=ingredient_embeddings)
else:
    logger.info(f"Vector space is up to date with {vector_count} vectors. No action needed.")