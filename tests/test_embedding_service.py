import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_with_cocktail_data import create_cocktails
from embedding_service import create_embeddings
from config.logger import logger

def test_create_embeddings():
    cocktails = create_cocktails()[:4]
    logger.info(cocktails)
    embedding_map = create_embeddings(cocktails=cocktails)
    logger.info(embedding_map)






if __name__ == "__main__":
    test_create_embeddings()