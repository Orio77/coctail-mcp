import sys
import os

# Add the parent directory to the path to import from the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.cocktail_data_service import create_cocktails

def test_create_cocktails():
    """Simple test that calls create_cocktails and prints the result"""
    cocktails = create_cocktails()
    print(f"Created {len(cocktails)} cocktails:")
    for cocktail in cocktails:
        print(f"- {cocktail.name}")

def test_create_ingredients():
    """Simple test that calls create_ingredients and prints the result"""
    from service.cocktail_data_service import create_ingredients
    ingredients = create_ingredients()
    print(f"Created {len(ingredients)} unique ingredients:")
    for ingredient in ingredients:
        print(f"- {ingredient.name}")

if __name__ == "__main__":
    test_create_cocktails()
    test_create_ingredients()