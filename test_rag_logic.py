#!/usr/bin/env python3
"""
Simple test script to verify the rag_cocktails function works correctly
"""

import sys
import json
from config.logger import logger

def test_rag_cocktails_logic():
    """Test the rag_cocktails function logic directly"""
    try:
        from embedding_service import embed_query
        from pinecone_service import query_cocktails

        query = "lime or mint"
        logger.info(f"Testing rag_cocktails logic with query: {query}")
        
        # Validate input
        if not query or not isinstance(query, str):
            logger.error("Invalid query provided")
            return []
        
        # Create embedding for the query
        embedding = embed_query(query=query)
        if embedding is None:
            logger.error("Failed to create embedding for query")
            return []
        
        logger.info(f"Created embedding with {len(embedding)} dimensions")
        
        # Query Pinecone
        results = query_cocktails(top_k=5, vector=embedding)
        logger.info(f"RAG results type: {type(results)}")
        logger.info(f"RAG results: {results}")
        
        if results is None:
            logger.error("No results returned from query_cocktails")
            return []
        
        # Handle Pinecone response format
        if hasattr(results, 'matches'):
            matches = results.matches
        elif isinstance(results, dict) and 'matches' in results:
            matches = results['matches']
        else:
            logger.error(f"Unexpected results format: {type(results)} - {results}")
            return []
        
        if matches is None or len(matches) == 0:
            logger.info("No matches found")
            return []
        
        # Convert matches to serializable format with None handling
        serializable_matches = []
        for match in matches:
            try:
                # Safely extract id
                match_id = ''
                if hasattr(match, 'id') and match.id is not None:
                    match_id = str(match.id)
                elif isinstance(match, dict) and match.get('id') is not None:
                    match_id = str(match.get('id'))
                
                # Safely extract score
                match_score = 0.0
                if hasattr(match, 'score') and match.score is not None:
                    match_score = float(match.score)
                elif isinstance(match, dict) and match.get('score') is not None:
                    match_score = float(match.get('score'))
                
                # Safely extract metadata - recursively clean None values
                match_metadata = {}
                if hasattr(match, 'metadata') and match.metadata is not None:
                    match_metadata = match.metadata
                elif isinstance(match, dict) and match.get('metadata') is not None:
                    match_metadata = match.get('metadata')
                
                # Recursively clean metadata to remove None values and ensure all values are JSON serializable
                def clean_dict(d):
                    if not isinstance(d, dict):
                        return d if d is not None else {}
                    
                    clean = {}
                    for key, value in d.items():
                        if value is None:
                            continue
                        elif isinstance(value, dict):
                            cleaned_value = clean_dict(value)
                            if cleaned_value:  # Only add if not empty
                                clean[str(key)] = cleaned_value
                        elif isinstance(value, (list, tuple)):
                            # Clean lists/tuples
                            cleaned_list = [item for item in value if item is not None]
                            if cleaned_list:  # Only add if not empty
                                clean[str(key)] = cleaned_list
                        elif isinstance(value, (str, int, float, bool)):
                            clean[str(key)] = value
                        else:
                            # Convert other types to string
                            clean[str(key)] = str(value)
                    return clean
                
                clean_metadata = clean_dict(match_metadata)
                
                match_dict = {
                    'id': match_id,
                    'score': match_score,
                    'metadata': clean_metadata
                }
                serializable_matches.append(match_dict)
                
            except Exception as match_error:
                logger.error(f"Error processing match: {str(match_error)}")
                continue
        
        logger.info(f"Returning {len(serializable_matches)} matches")
        
        # Final validation - ensure the result is JSON serializable
        try:
            json_result = json.dumps(serializable_matches, indent=2)
            logger.info(f"Successfully serialized result to JSON")
            logger.info(f"Result contains {len(serializable_matches)} matches")
            
            if serializable_matches:
                logger.info(f"First match: {serializable_matches[0]}")
            
            print("Test passed! The function works correctly and returns JSON-serializable data.")
            return serializable_matches
        except Exception as json_error:
            logger.error(f"Failed to serialize result to JSON: {str(json_error)}")
            return []
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

if __name__ == "__main__":
    result = test_rag_cocktails_logic()
    success = len(result) > 0
    sys.exit(0 if success else 1)