import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinecone_service import upsert_embeddings, clear_vectorspace, query_cocktails
from work_with_cocktail_data import create_cocktails
from embedding_service import create_embeddings, embed_query
from config.logger import logger

def test_upsert_embeddings():
    cocktails = create_cocktails()
    embeddings = create_embeddings(cocktails=cocktails)

    upsert_embeddings(cocktails=cocktails, embeddings=embeddings)
    # clear_vectorspace()

def test_query_cocktails():
    query = "A refreshing cocktail with lime and mint"
    embedding = embed_query(query=query)

    res = query_cocktails(top_k=5, vector=embedding)
    logger.info(res)

if __name__ == "__main__":
    test_upsert_embeddings()
    # test_query_cocktails()