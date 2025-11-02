#!/usr/bin/env python3
"""
Simple test script to verify the rag_cocktails function works correctly
"""

import sys
import json
from config.logger import logger

def test_rag_cocktails():
    """Test the rag_cocktails function directly"""
    try:
        # Import the function
        from cocktail_server import rag_cocktails
        
        # Test with a simple query
        query = "lime or mint"
        logger.info(f"Testing rag_cocktails with query: {query}")
        
        result = rag_cocktails(query)
        
        # Verify the result
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        
        # Try to serialize to JSON
        json_result = json.dumps(result, indent=2)
        logger.info(f"Successfully serialized result to JSON")
        logger.info(f"Result contains {len(result)} matches")
        
        if result:
            logger.info(f"First match: {result[0]}")
        
        print("✅ Test passed! The function works correctly and returns JSON-serializable data.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_rag_cocktails()
    sys.exit(0 if success else 1)