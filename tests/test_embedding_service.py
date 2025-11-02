import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_with_cocktail_data import create_cocktails
from embedding_service import create_embeddings, embed_query
from config.logger import logger

def test_create_embeddings():
    cocktails = create_cocktails()[:4]
    logger.info(cocktails)
    embedding_map = create_embeddings(cocktails=cocktails)
    logger.info(embedding_map)

def test_embed_query():
    query = "A refreshing cocktail with lime and mint"
    embedding = embed_query(query=query)
    logger.info(embedding[:30])





if __name__ == "__main__":
    # test_create_embeddings()
    test_embed_query()