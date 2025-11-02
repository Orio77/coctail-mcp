import json
import traceback
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

from config.logger import logger
from services.embedding_service import embed_query
from services.pinecone_service import query_cocktails


@dataclass
class RAGMatch:
    """Represents a single match from the RAG system."""
    id: str
    score: float
    metadata: Dict[str, Any]


class RAGError(Exception):
    """Custom exception for RAG-related errors."""
    pass


def run_cocktail_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve cocktail recommendations based on a query.
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 5)
        
    Returns:
        List of cocktail matches with id, score, and metadata
        
    Raises:
        RAGError: If the RAG process fails
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Validate input
        _validate_query(query)
        
        # Create embedding for the query
        embedding = _create_query_embedding(query)
        
        # Query vector database
        results = _query_vector_database(embedding, top_k)
        
        # Process and serialize matches
        processed_matches = _process_query_results(results)
        
        logger.info(f"Successfully processed {len(processed_matches)} matches")
        return processed_matches
        
    except RAGError:
        # Re-raise RAG-specific errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_cocktail_rag: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RAGError(f"RAG processing failed: {str(e)}") from e


def _validate_query(query: str) -> None:
    """Validate the input query."""
    if not query or not isinstance(query, str) or not query.strip():
        raise RAGError("Query must be a non-empty string")


def _create_query_embedding(query: str) -> List[float]:
    """Create embedding for the query."""
    try:
        embedding = embed_query(query=query.strip())
        if embedding is None:
            raise RAGError("Failed to create embedding for query")
        
        logger.info(f"Created embedding with {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        raise RAGError(f"Embedding creation failed: {str(e)}") from e


def _query_vector_database(embedding: List[float], top_k: int) -> Any:
    """Query the vector database with the embedding."""
    try:
        results = query_cocktails(top_k=top_k, vector=embedding)
        if results is None:
            raise RAGError("No results returned from vector database")
        
        logger.debug(f"Vector database results type: {type(results)}")
        return results
    except Exception as e:
        logger.error(f"Vector database query failed: {str(e)}")
        raise RAGError(f"Vector database query failed: {str(e)}") from e


def _process_query_results(results: Any) -> List[Dict[str, Any]]:
    """Process and serialize query results."""
    # Extract matches from results
    matches = _extract_matches(results)
    
    if not matches:
        logger.info("No matches found")
        return []
    
    # Convert matches to serializable format
    processed_matches = []
    for match in matches:
        try:
            processed_match = _process_single_match(match)
            processed_matches.append(processed_match)
        except Exception as match_error:
            logger.warning(f"Skipping match due to processing error: {str(match_error)}")
            continue
    
    # Validate JSON serialization
    _validate_json_serialization(processed_matches)
    
    return processed_matches


def _extract_matches(results: Any) -> List[Any]:
    """Extract matches from the results object."""
    if hasattr(results, 'matches'):
        return results.matches or []
    elif isinstance(results, dict) and 'matches' in results:
        return results['matches'] or []
    else:
        raise RAGError(f"Unexpected results format: {type(results)}")


def _process_single_match(match: Any) -> Dict[str, Any]:
    """Process a single match into a serializable format."""
    return {
        'id': _extract_match_id(match),
        'score': _extract_match_score(match),
        'metadata': _extract_and_clean_metadata(match)
    }


def _extract_match_id(match: Any) -> str:
    """Safely extract match ID."""
    if hasattr(match, 'id') and match.id is not None:
        return str(match.id)
    elif isinstance(match, dict) and match.get('id') is not None:
        return str(match.get('id'))
    else:
        return ''


def _extract_match_score(match: Any) -> float:
    """Safely extract match score."""
    if hasattr(match, 'score') and match.score is not None:
        return float(match.score)
    elif isinstance(match, dict) and match.get('score') is not None:
        return float(match.get('score'))
    else:
        return 0.0


def _extract_and_clean_metadata(match: Any) -> Dict[str, Any]:
    """Safely extract and clean match metadata."""
    metadata = {}
    if hasattr(match, 'metadata') and match.metadata is not None:
        metadata = match.metadata
    elif isinstance(match, dict) and match.get('metadata') is not None:
        metadata = match.get('metadata')
    
    return _clean_metadata_dict(metadata)


def _clean_metadata_dict(data: Any) -> Dict[str, Any]:
    """
    Recursively clean metadata to remove None values and ensure JSON serialization.
    
    Args:
        data: The data to clean
        
    Returns:
        Cleaned dictionary with None values removed and all values serializable
    """
    if not isinstance(data, dict):
        return {} if data is None else _serialize_value(data)
    
    cleaned = {}
    for key, value in data.items():
        if value is None:
            continue
        
        cleaned_value = _clean_value(value)
        if cleaned_value is not None and cleaned_value != {}:
            cleaned[str(key)] = cleaned_value
    
    return cleaned


def _clean_value(value: Any) -> Any:
    """Clean a single value for JSON serialization."""
    if value is None:
        return None
    elif isinstance(value, dict):
        return _clean_metadata_dict(value)
    elif isinstance(value, (list, tuple)):
        cleaned_list = [_clean_value(item) for item in value if item is not None]
        return [item for item in cleaned_list if item is not None]
    else:
        return _serialize_value(value)


def _serialize_value(value: Any) -> Union[str, int, float, bool]:
    """Convert a value to a JSON-serializable type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    else:
        return str(value)


def _validate_json_serialization(data: List[Dict[str, Any]]) -> None:
    """Validate that the data can be JSON serialized."""
    try:
        json.dumps(data)
        logger.debug("JSON serialization validation passed")
    except (TypeError, ValueError) as e:
        raise RAGError(f"Failed to serialize result to JSON: {str(e)}") from e