import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service.pinecone_service import upsert_cocktail_embeddings, clear_vectorspace, query_cocktails
from service.cocktail_data_service import create_cocktails
from service.embedding_service import create_embeddings, embed_query
from config.logger import logger

def test_upsert_embeddings():
    cocktails = create_cocktails()
    embeddings = create_embeddings(cocktails=cocktails)

    upsert_cocktail_embeddings(cocktails=cocktails, embeddings=embeddings)
    # clear_vectorspace()

def test_query_cocktails():
    query = "A refreshing cocktail with lime and mint"
    embedding = embed_query(query=query)

    res = query_cocktails(top_k=5, vector=embedding)
    logger.info(res)

if __name__ == "__main__":
    test_upsert_embeddings()
    # test_query_cocktails()