import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logger import logger
from service.cocktail_data_service import create_cocktails
from service.embedding_service import create_cocktail_embeddings, embed_query

dimensions: int = int(os.getenv('embedding_dimension'))
logger.info(f"Embedding dimension from environment: {dimensions}")

@pytest.fixture
def sample_cocktails():
    """Fixture to provide a small sample of cocktails for testing."""
    logger.info("Loading sample cocktails for testing...")
    return create_cocktails()[:4]

def test_create_embeddings(sample_cocktails):
    """Tests the creation of cocktail embeddings."""
    # Create embeddings
    embedding_map = create_cocktail_embeddings(cocktails=sample_cocktails)
    
    # Verify embeddings map
    assert isinstance(embedding_map, dict)
    # Check that all cocktail IDs are present
    assert len(embedding_map) == len(sample_cocktails)
    # Check that embeddings are numeric vectors
    for embedding in embedding_map.values():
        assert isinstance(embedding, (list, tuple))
        assert len(embedding) > 0

@pytest.mark.parametrize("query,expected_length", [
    ("A refreshing cocktail with lime and mint", dimensions),  
    ("Strong whiskey drink", dimensions),
    ("Sweet fruity cocktail", dimensions),
])
def test_embed_query(query, expected_length):
    """Tests embedding generation for queries."""
    embedding = embed_query(query=query)
    
    assert isinstance(embedding, (list, tuple))
    assert len(embedding) == expected_length
    # Check that embedding contains numeric values
    assert all(isinstance(x, (int, float)) for x in embedding[:10])

def test_embed_query_empty_string():
    """Tests embedding generation for empty query."""
    with pytest.raises(ValueError):
        embed_query("")