import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinecone_service import upsert_embeddings, clear_vectorspace
from work_with_cocktail_data import create_cocktails
from embedding_service import create_embeddings

def test_upsert_embeddings():
    cocktails = create_cocktails()[:3]
    embeddings = create_embeddings(cocktails=cocktails)

    upsert_embeddings(cocktails=cocktails, embeddings=embeddings)
    clear_vectorspace()


if __name__ == "__main__":
    test_upsert_embeddings()