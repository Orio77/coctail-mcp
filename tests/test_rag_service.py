import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rag_service import (
    run_cocktail_rag, 
    RAGError, 
    _validate_query,
    _create_query_embedding,
    _query_vector_database,
    _process_query_results,
    _extract_matches,
    _process_single_match,
    _clean_metadata_dict,
    _serialize_value
)


class TestRunCocktailRAG:
    """Test cases for run_cocktail_rag function"""
    
    def test_successful_rag_query(self):
        """Test successful RAG query execution"""
        mock_embedding = [0.1, 0.2, 0.3]
        mock_results = MagicMock()
        mock_results.matches = [
            MagicMock(
                id="cocktail_1",
                score=0.95,
                metadata={"name": "Mojito", "type": "cocktail"}
            )
        ]
        
        with patch('services.rag_service.embed_query', return_value=mock_embedding):
            with patch('services.rag_service.query_cocktails', return_value=mock_results):
                result = run_cocktail_rag("refreshing cocktail", top_k=5)
                
                assert len(result) == 1
                assert result[0]['id'] == "cocktail_1"
                assert result[0]['score'] == 0.95
                assert result[0]['metadata']['name'] == "Mojito"
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises RAGError"""
        with pytest.raises(RAGError, match="Query must be a non-empty string"):
            run_cocktail_rag("")
    
    def test_none_query_raises_error(self):
        """Test that None query raises RAGError"""
        with pytest.raises(RAGError, match="Query must be a non-empty string"):
            run_cocktail_rag(None)
    
    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises RAGError"""
        with pytest.raises(RAGError, match="Query must be a non-empty string"):
            run_cocktail_rag("   ")
    
    def test_embedding_failure_raises_error(self):
        """Test that embedding failure raises RAGError"""
        with patch('services.rag_service.embed_query', return_value=None):
            with pytest.raises(RAGError, match="Failed to create embedding for query"):
                run_cocktail_rag("test query")
    
    def test_vector_db_failure_raises_error(self):
        """Test that vector database failure raises RAGError"""
        with patch('services.rag_service.embed_query', return_value=[0.1, 0.2]):
            with patch('services.rag_service.query_cocktails', return_value=None):
                with pytest.raises(RAGError, match="No results returned from vector database"):
                    run_cocktail_rag("test query")


class TestValidateQuery:
    """Test cases for _validate_query function"""
    
    def test_valid_query(self):
        """Test that valid query passes validation"""
        _validate_query("valid query")  # Should not raise
    
    def test_empty_string_raises_error(self):
        """Test that empty string raises error"""
        with pytest.raises(RAGError):
            _validate_query("")
    
    def test_none_raises_error(self):
        """Test that None raises error"""
        with pytest.raises(RAGError):
            _validate_query(None)
    
    def test_whitespace_raises_error(self):
        """Test that whitespace-only string raises error"""
        with pytest.raises(RAGError):
            _validate_query("   ")


class TestCreateQueryEmbedding:
    """Test cases for _create_query_embedding function"""
    
    def test_successful_embedding_creation(self):
        """Test successful embedding creation"""
        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        with patch('services.rag_service.embed_query', return_value=mock_embedding):
            result = _create_query_embedding("test query")
            assert result == mock_embedding
    
    def test_none_embedding_raises_error(self):
        """Test that None embedding raises error"""
        with patch('services.rag_service.embed_query', return_value=None):
            with pytest.raises(RAGError, match="Failed to create embedding for query"):
                _create_query_embedding("test query")
    
    def test_embedding_service_exception_raises_rag_error(self):
        """Test that embedding service exception raises RAGError"""
        with patch('services.rag_service.embed_query', side_effect=Exception("Embedding failed")):
            with pytest.raises(RAGError, match="Embedding creation failed"):
                _create_query_embedding("test query")


class TestQueryVectorDatabase:
    """Test cases for _query_vector_database function"""
    
    def test_successful_query(self):
        """Test successful vector database query"""
        mock_results = MagicMock()
        with patch('services.rag_service.query_cocktails', return_value=mock_results):
            result = _query_vector_database([0.1, 0.2], top_k=5)
            assert result == mock_results
    
    def test_none_results_raises_error(self):
        """Test that None results raise error"""
        with patch('services.rag_service.query_cocktails', return_value=None):
            with pytest.raises(RAGError, match="No results returned from vector database"):
                _query_vector_database([0.1, 0.2], top_k=5)
    
    def test_query_exception_raises_rag_error(self):
        """Test that query exception raises RAGError"""
        with patch('services.rag_service.query_cocktails', side_effect=Exception("Query failed")):
            with pytest.raises(RAGError, match="Vector database query failed"):
                _query_vector_database([0.1, 0.2], top_k=5)


class TestExtractMatches:
    """Test cases for _extract_matches function"""
    
    def test_extract_matches_from_object_with_matches_attribute(self):
        """Test extracting matches from object with matches attribute"""
        mock_results = MagicMock()
        mock_results.matches = ["match1", "match2"]
        result = _extract_matches(mock_results)
        assert result == ["match1", "match2"]
    
    def test_extract_matches_from_dict(self):
        """Test extracting matches from dictionary"""
        mock_results = {"matches": ["match1", "match2"]}
        result = _extract_matches(mock_results)
        assert result == ["match1", "match2"]
    
    def test_extract_matches_empty_list(self):
        """Test extracting empty matches list"""
        mock_results = MagicMock()
        mock_results.matches = []
        result = _extract_matches(mock_results)
        assert result == []
    
    def test_extract_matches_none_matches(self):
        """Test extracting None matches"""
        mock_results = MagicMock()
        mock_results.matches = None
        result = _extract_matches(mock_results)
        assert result == []
    
    def test_extract_matches_invalid_format_raises_error(self):
        """Test that invalid format raises error"""
        with pytest.raises(RAGError, match="Unexpected results format"):
            _extract_matches("invalid")


class TestProcessSingleMatch:
    """Test cases for _process_single_match function"""
    
    def test_process_match_object(self):
        """Test processing match object with attributes"""
        mock_match = MagicMock()
        mock_match.id = "cocktail_1"
        mock_match.score = 0.95
        mock_match.metadata = {"name": "Mojito"}
        
        result = _process_single_match(mock_match)
        assert result['id'] == "cocktail_1"
        assert result['score'] == 0.95
        assert result['metadata']['name'] == "Mojito"
    
    def test_process_match_dict(self):
        """Test processing match dictionary"""
        mock_match = {
            "id": "cocktail_2",
            "score": 0.87,
            "metadata": {"name": "Margarita", "type": "cocktail"}
        }
        
        result = _process_single_match(mock_match)
        assert result['id'] == "cocktail_2"
        assert result['score'] == 0.87
        assert result['metadata']['name'] == "Margarita"
    
    def test_process_match_missing_fields(self):
        """Test processing match with missing fields"""
        mock_match = MagicMock()
        mock_match.id = None
        mock_match.score = None
        mock_match.metadata = None
        
        result = _process_single_match(mock_match)
        assert result['id'] == ""
        assert result['score'] == 0.0
        assert result['metadata'] == {}


class TestCleanMetadataDict:
    """Test cases for _clean_metadata_dict function"""
    
    def test_clean_valid_metadata(self):
        """Test cleaning valid metadata"""
        metadata = {
            "name": "Mojito",
            "type": "cocktail",
            "ingredients": ["rum", "mint"]
        }
        result = _clean_metadata_dict(metadata)
        assert result == metadata
    
    def test_clean_metadata_with_none_values(self):
        """Test cleaning metadata with None values"""
        metadata = {
            "name": "Mojito",
            "description": None,
            "type": "cocktail",
            "image": None
        }
        result = _clean_metadata_dict(metadata)
        assert result == {"name": "Mojito", "type": "cocktail"}
    
    def test_clean_nested_metadata(self):
        """Test cleaning nested metadata"""
        metadata = {
            "cocktail": {
                "name": "Mojito",
                "details": None
            },
            "ingredients": ["rum", None, "mint"]
        }
        result = _clean_metadata_dict(metadata)
        assert result["cocktail"]["name"] == "Mojito"
        assert "details" not in result["cocktail"]
        assert result["ingredients"] == ["rum", "mint"]
    
    def test_clean_non_dict_input(self):
        """Test cleaning non-dict input"""
        result = _clean_metadata_dict("not a dict")
        assert result == "not a dict"
        
        result = _clean_metadata_dict(None)
        assert result == {}


class TestSerializeValue:
    """Test cases for _serialize_value function"""
    
    def test_serialize_string(self):
        """Test serializing string value"""
        assert _serialize_value("test") == "test"
    
    def test_serialize_int(self):
        """Test serializing int value"""
        assert _serialize_value(42) == 42
    
    def test_serialize_float(self):
        """Test serializing float value"""
        assert _serialize_value(3.14) == 3.14
    
    def test_serialize_bool(self):
        """Test serializing bool value"""
        assert _serialize_value(True) == True
        assert _serialize_value(False) == False
    
    def test_serialize_object_to_string(self):
        """Test serializing complex object to string"""
        class TestObj:
            def __str__(self):
                return "test_object"
        
        obj = TestObj()
        assert _serialize_value(obj) == "test_object"


class TestProcessQueryResults:
    """Test cases for _process_query_results function"""
    
    def test_process_valid_results(self):
        """Test processing valid query results"""
        mock_results = MagicMock()
        mock_match = MagicMock()
        mock_match.id = "cocktail_1"
        mock_match.score = 0.95
        mock_match.metadata = {"name": "Mojito"}
        mock_results.matches = [mock_match]
        
        result = _process_query_results(mock_results)
        assert len(result) == 1
        assert result[0]['id'] == "cocktail_1"
    
    def test_process_empty_results(self):
        """Test processing empty results"""
        mock_results = MagicMock()
        mock_results.matches = []
        
        result = _process_query_results(mock_results)
        assert result == []
    
    def test_process_results_with_invalid_match(self):
        """Test processing results with one invalid match"""
        mock_results = MagicMock()
        valid_match = MagicMock()
        valid_match.id = "cocktail_1"
        valid_match.score = 0.95
        valid_match.metadata = {"name": "Mojito"}
        
        # Create an invalid match that will cause an exception
        invalid_match = MagicMock()
        invalid_match.id = None
        invalid_match.score = "invalid_score"  # This will cause an error in float conversion
        
        mock_results.matches = [valid_match, invalid_match]
        
        # Mock _process_single_match to raise exception for invalid match
        with patch('services.rag_service._process_single_match') as mock_process:
            mock_process.side_effect = [
                {"id": "cocktail_1", "score": 0.95, "metadata": {"name": "Mojito"}},
                Exception("Invalid match")
            ]
            
            result = _process_query_results(mock_results)
            assert len(result) == 1
            assert result[0]['id'] == "cocktail_1"


# Simple integration tests
def test_run_cocktail_rag_integration():
    """Integration test for run_cocktail_rag"""
    mock_embedding = [0.1, 0.2, 0.3]
    mock_results = MagicMock()
    mock_results.matches = [
        MagicMock(
            id="cocktail_1",
            score=0.95,
            metadata={"name": "Mojito", "type": "cocktail"}
        )
    ]
    
    with patch('services.rag_service.embed_query', return_value=mock_embedding):
        with patch('services.rag_service.query_cocktails', return_value=mock_results):
            result = run_cocktail_rag("refreshing mint cocktail")
            print(f"Found {len(result)} cocktail matches")
            for match in result:
                print(f"- {match['metadata'].get('name', 'Unknown')} (score: {match['score']})")


if __name__ == "__main__":
    test_run_cocktail_rag_integration()