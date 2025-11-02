import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pinecone_service import (
    upsert_cocktail_embeddings, 
    upsert_ingredient_embeddings,
    clear_vectorspace, 
    query_cocktails,
    get_num_vectors,
    PineconeError,
    _create_cocktail_vectors_to_upsert,
    _create_ingredient_vectors_to_upsert,
    _is_valid_cocktail,
    _is_valid_ingredient,
    _safe_string,
    _safe_list,
    _safe_bool
)
from models.cocktail import Cocktail
from models.ingredient import Ingredient


class TestPineconeService:
    
    @patch('services.pinecone_service.index')
    def test_upsert_cocktail_embeddings_success(self, mock_index):
        """Test successful cocktail embedding upsert."""
        cocktails = [Mock(id=1, name="Mojito", tags=["refreshing"], instructions="Mix", image_url="url", ingredients=["mint"])]
        embeddings = {1: [0.1, 0.2, 0.3]}
        
        upsert_cocktail_embeddings(cocktails, embeddings)
        mock_index.upsert.assert_called_once()

    def test_upsert_cocktail_embeddings_empty_cocktails(self):
        """Test upsert with empty cocktails list."""
        upsert_cocktail_embeddings([], {1: [0.1, 0.2]})
        # Should not raise exception

    def test_upsert_cocktail_embeddings_empty_embeddings(self):
        """Test upsert with empty embeddings dict."""
        cocktails = [Mock(id=1)]
        upsert_cocktail_embeddings(cocktails, {})
        # Should not raise exception

    @patch('services.pinecone_service.index')
    def test_upsert_ingredient_embeddings_success(self, mock_index):
        """Test successful ingredient embedding upsert."""
        ingredients = [Mock(id=1, name="Vodka", description="Spirit", alcohol=True, type="Spirit", image_url="url")]
        embeddings = {1: [0.1, 0.2, 0.3]}
        
        upsert_ingredient_embeddings(ingredients, embeddings)
        mock_index.upsert.assert_called_once()

    @patch('services.pinecone_service.index')
    def test_clear_vectorspace_success(self, mock_index):
        """Test successful vector space clearing."""
        mock_index.delete.return_value = {"deleted": True}
        
        clear_vectorspace()
        mock_index.delete.assert_called_once_with(delete_all=True)

    @patch('services.pinecone_service.index')
    def test_clear_vectorspace_failure(self, mock_index):
        """Test vector space clearing failure."""
        mock_index.delete.side_effect = Exception("Connection error")
        
        with pytest.raises(PineconeError):
            clear_vectorspace()

    @patch('services.pinecone_service.index')
    def test_query_cocktails_success(self, mock_index):
        """Test successful cocktail query."""
        mock_result = Mock()
        mock_result.matches = [Mock(), Mock()]
        mock_index.query.return_value = mock_result
        
        result = query_cocktails(5, [0.1, 0.2, 0.3])
        
        assert result == mock_result
        mock_index.query.assert_called_once_with(top_k=5, include_metadata=True, vector=[0.1, 0.2, 0.3])

    def test_query_cocktails_empty_vector(self):
        """Test query with empty vector."""
        result = query_cocktails(5, [])
        assert result is None

    def test_query_cocktails_invalid_top_k(self):
        """Test query with invalid top_k values."""
        with patch('services.pinecone_service.index') as mock_index:
            # Create a proper mock result with matches attribute
            mock_result = Mock()
            mock_result.matches = []
            mock_index.query.return_value = mock_result
            
            # Test negative top_k
            query_cocktails(-1, [0.1, 0.2])
            mock_index.query.assert_called_with(top_k=5, include_metadata=True, vector=[0.1, 0.2])
            
            # Test too large top_k
            query_cocktails(2000, [0.1, 0.2])
            mock_index.query.assert_called_with(top_k=1000, include_metadata=True, vector=[0.1, 0.2])

    @patch('services.pinecone_service.index')
    def test_get_num_vectors_success(self, mock_index):
        """Test successful vector count retrieval."""
        mock_index.describe_index_stats.return_value = {"total_vector_count": 100}
        
        count = get_num_vectors()
        assert count == 100

    @patch('services.pinecone_service.index')
    def test_get_num_vectors_failure(self, mock_index):
        """Test vector count retrieval failure."""
        mock_index.describe_index_stats.side_effect = Exception("API error")
        
        with pytest.raises(PineconeError):
            get_num_vectors()

    def test_create_cocktail_vectors_to_upsert(self):
        """Test cocktail vector creation."""
        cocktail_mock = Mock()
        cocktail_mock.id = 1
        cocktail_mock.name = "Mojito"
        cocktail_mock.tags = ["fresh"]
        cocktail_mock.instructions = "Mix"
        cocktail_mock.image_url = "url"
        cocktail_mock.ingredients = ["mint"]
        
        cocktails = [cocktail_mock]
        embeddings = {1: [0.1, 0.2, 0.3]}
        
        vectors = _create_cocktail_vectors_to_upsert(cocktails, embeddings)
        
        assert len(vectors) == 1
        assert vectors[0]["id"] == "cocktail_1"
        assert vectors[0]["values"] == [0.1, 0.2, 0.3]
        assert vectors[0]["metadata"]["type"] == "cocktail"
        assert vectors[0]["metadata"]["name"] == "Mojito"

    def test_create_ingredient_vectors_to_upsert(self):
        """Test ingredient vector creation."""
        ingredient_mock = Mock()
        ingredient_mock.id = 1
        ingredient_mock.name = "Vodka"
        ingredient_mock.description = "Spirit"
        ingredient_mock.alcohol = True
        ingredient_mock.type = "Spirit"
        ingredient_mock.image_url = "url"
        
        ingredients = [ingredient_mock]
        embeddings = {1: [0.1, 0.2, 0.3]}
        
        vectors = _create_ingredient_vectors_to_upsert(ingredients, embeddings)
        
        assert len(vectors) == 1
        assert vectors[0]["id"] == "ingredient_1"
        assert vectors[0]["values"] == [0.1, 0.2, 0.3]
        assert vectors[0]["metadata"]["type"] == "ingredient"
        assert vectors[0]["metadata"]["name"] == "Vodka"

    def test_is_valid_cocktail(self):
        """Test cocktail validation."""
        valid_cocktail = Mock(id=1)
        invalid_cocktail_no_id = Mock()
        delattr(invalid_cocktail_no_id, 'id')
        
        assert _is_valid_cocktail(valid_cocktail) == True
        assert _is_valid_cocktail(invalid_cocktail_no_id) == False
        assert _is_valid_cocktail(None) == False

    def test_is_valid_ingredient(self):
        """Test ingredient validation."""
        valid_ingredient = Mock(id=1)
        invalid_ingredient_no_id = Mock()
        delattr(invalid_ingredient_no_id, 'id')
        
        assert _is_valid_ingredient(valid_ingredient) == True
        assert _is_valid_ingredient(invalid_ingredient_no_id) == False
        assert _is_valid_ingredient(None) == False

    def test_safe_string(self):
        """Test safe string conversion."""
        assert _safe_string(None) == ""
        assert _safe_string("test") == "test"
        assert _safe_string(123) == "123"
        assert _safe_string(True) == "True"

    def test_safe_list(self):
        """Test safe list conversion."""
        assert _safe_list(None) == []
        assert _safe_list([1, 2, 3]) == [1, 2, 3]
        assert _safe_list((1, 2)) == [1, 2]
        assert _safe_list({1, 2}) == [1, 2]
        assert _safe_list("test") == ["test"]

    def test_safe_bool(self):
        """Test safe boolean conversion."""
        assert _safe_bool(None) == False
        assert _safe_bool(True) == True
        assert _safe_bool(False) == False
        assert _safe_bool("true") == True
        assert _safe_bool("false") == False
        assert _safe_bool("1") == True
        assert _safe_bool("0") == False
        assert _safe_bool(1) == True
        assert _safe_bool(0) == False