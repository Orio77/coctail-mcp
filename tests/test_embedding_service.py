import pytest
import sys
import os
from unittest.mock import patch, Mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logger import logger
from services.cocktail_data_service import create_cocktails
from services.embedding_service import (
    create_cocktail_embeddings, 
    create_ingredient_embeddings,
    embed_query,
    _get_embedding_model,
    _create_embedding
)
from models.cocktail import Cocktail
from models.ingredient import Ingredient

dimensions: int = int(os.getenv('EMBEDDING_DIMENSION'))
logger.info(f"Embedding dimension from environment: {dimensions}")

@pytest.fixture
def sample_cocktails():
    """Fixture to provide a small sample of cocktails for testing."""
    logger.info("Loading sample cocktails for testing...")
    return create_cocktails()[:4]

@pytest.fixture
def sample_ingredients():
    """Fixture to provide sample ingredients for testing."""
    return [
        Ingredient(
            id=1,
            name="Vodka",
            description="A clear distilled alcoholic beverage.",
            alcohol=True,
            type="Spirit",
            image_url="https://example.com/vodka.png"
        ),
        Ingredient(
            id=2,
            name="Lime Juice",
            description="Juice squeezed from fresh limes.",
            alcohol=False,
            type="Mixer",
            image_url="https://example.com/lime_juice.png"
        ),
        Ingredient(
            id=3,
            name="Simple Syrup",
            description="A sweetener made by dissolving sugar in water.",
            alcohol=False,
            type="Sweetener",
            image_url="https://example.com/simple_syrup.png"
        ),
    ]

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

def test_embed_query_whitespace_only():
    """Tests embedding generation for whitespace-only query."""
    with pytest.raises(ValueError):
        embed_query("   \t\n   ")

def test_create_cocktail_embeddings_empty_list():
    """Tests error handling for empty cocktail list."""
    with pytest.raises(ValueError, match="Cocktails list cannot be empty"):
        create_cocktail_embeddings([])

def test_create_ingredient_embeddings(sample_ingredients):
    """Tests the creation of ingredient embeddings."""
    embedding_map = create_ingredient_embeddings(ingredients=sample_ingredients)
    
    assert isinstance(embedding_map, dict)
    assert len(embedding_map) == len(sample_ingredients)
    for ingredient_id, embedding in embedding_map.items():
        assert ingredient_id in [ing.id for ing in sample_ingredients]
        assert isinstance(embedding, (list, tuple))
        assert len(embedding) > 0

def test_create_ingredient_embeddings_empty_list():
    """Tests error handling for empty ingredient list."""
    with pytest.raises(ValueError, match="Ingredients list cannot be empty"):
        create_ingredient_embeddings([])

@patch.dict(os.environ, {}, clear=True)
def test_get_embedding_model_missing_env_var():
    """Tests error handling when embedding model env var is not set."""
    with pytest.raises(ValueError, match="EMBEDDING_MODEL environment variable is not set"):
        _get_embedding_model()

@patch('services.embedding_service.ollama.embed')
def test_create_embedding_no_embeddings_returned(mock_embed):
    """Tests error handling when no embeddings are returned from model."""
    mock_embed.return_value = Mock(embeddings=[])
    
    with pytest.raises(ValueError, match="No embedding returned from model"):
        _create_embedding("test_model", "test input")

@patch('services.embedding_service.ollama.embed')
def test_create_embedding_ollama_exception(mock_embed):
    """Tests error handling when ollama throws an exception."""
    mock_embed.side_effect = Exception("Ollama connection error")
    
    with pytest.raises(Exception):
        _create_embedding("test_model", "test input")

@patch('services.embedding_service._create_embedding')
def test_create_cocktail_embeddings_partial_failure(mock_create_embedding, sample_cocktails):
    """Tests that cocktail embedding creation continues when some cocktails fail."""
    # Mock first cocktail to fail, others succeed
    mock_create_embedding.side_effect = [
        Exception("Failed"),
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    
    embedding_map = create_cocktail_embeddings(sample_cocktails)
    
    # Should have 3 embeddings (one failed)
    assert len(embedding_map) == 3
    assert sample_cocktails[0].id not in embedding_map

@patch('services.embedding_service._create_embedding')
def test_create_cocktail_embeddings_all_fail(mock_create_embedding, sample_cocktails):
    """Tests error handling when all cocktail embeddings fail."""
    mock_create_embedding.side_effect = Exception("All failed")
    
    with pytest.raises(RuntimeError, match="Failed to create any cocktail embeddings"):
        create_cocktail_embeddings(sample_cocktails)

@patch('services.embedding_service._create_embedding')
def test_create_ingredient_embeddings_partial_failure(mock_create_embedding, sample_ingredients):
    """Tests that ingredient embedding creation continues when some ingredients fail."""
    mock_create_embedding.side_effect = [
        Exception("Failed"),
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    
    embedding_map = create_ingredient_embeddings(sample_ingredients)
    
    assert len(embedding_map) == 2
    assert sample_ingredients[0].id not in embedding_map

@patch('services.embedding_service._create_embedding')
def test_create_ingredient_embeddings_all_fail(mock_create_embedding, sample_ingredients):
    """Tests error handling when all ingredient embeddings fail."""
    mock_create_embedding.side_effect = Exception("All failed")
    
    with pytest.raises(RuntimeError, match="Failed to create any ingredient embeddings"):
        create_ingredient_embeddings(sample_ingredients)

@patch('services.embedding_service._get_embedding_model')
@patch('services.embedding_service._create_embedding')
def test_embed_query_strips_whitespace(mock_create_embedding, mock_get_model):
    """Tests that embed_query strips whitespace from query."""
    mock_get_model.return_value = "test_model"
    mock_create_embedding.return_value = [1.0, 2.0, 3.0]
    
    result = embed_query("  test query  ")
    
    mock_create_embedding.assert_called_once_with("test_model", "test query")
    assert result == [1.0, 2.0, 3.0]

@patch('services.embedding_service.ollama.embed')
def test_create_embedding_float_conversion(mock_embed):
    """Tests that embedding values are properly converted to floats."""
    mock_embed.return_value = Mock(embeddings=[[1, 2.5, 3]])
    
    result = _create_embedding("test_model", "test input")
    
    assert result == [1.0, 2.5, 3.0]
    assert all(isinstance(x, float) for x in result)