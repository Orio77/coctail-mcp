from pinecone import Pinecone
import os
import dotenv
from models.cocktail import Cocktail
from config.logger import logger

dotenv.load_dotenv()

pc = Pinecone()
index = pc.Index(os.getenv('pinecone_index_name'))

def upsert_embeddings(cocktails: list[Cocktail], embeddings: dict[int, list[float]]):
    vectors = _create_vectors_to_upsert(cocktails=cocktails, embeddings=embeddings)
    logger.info(vectors)

    response = index.upsert(vectors=vectors)
    logger.info(response)


def clear_vectorspace():
    index.delete(delete_all=True)



def _create_vectors_to_upsert(cocktails: list[Cocktail], embeddings: dict[int, list[float]]):
    vectors = []

    for cocktail in cocktails:
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

    return vectors