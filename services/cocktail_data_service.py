import dotenv
import os
import json

from models.cocktail import Cocktail
from models.ingredient import Ingredient

dotenv.load_dotenv()

def load_cocktail_data():
    data_path = os.getenv("DATA_PATH")
    
    # Handle missing environment variable
    if not data_path:
        raise ValueError("data_path environment variable is not set")
    
    # Handle file not found
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at path: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading data file: {e}")
    
    # Handle empty or invalid data
    if not isinstance(data, list):
        raise ValueError("Data file must contain a list of cocktails")
    
    return data

# function to create cocktail objects from data file
def create_cocktails():
    """Creates and returns a list of Cocktail objects from the data file."""
    # read cocktails from file
    data = load_cocktail_data()

    # prepare list for cocktails
    cocktail_list = []

    # create cocktail objects from data rows
    for row in data:
        # Skip invalid rows
        if not isinstance(row, dict) or 'id' not in row or 'name' not in row:
            continue
            
        # Handle missing ingredients
        ingredients_data = row.get('ingredients', [])
        ingredient_names = []
        
        if isinstance(ingredients_data, list):
            ingredient_names = [ingredient.get('name', '') for ingredient in ingredients_data 
                             if isinstance(ingredient, dict) and ingredient.get('name')]

        cocktail = Cocktail(
            id=row['id'],
            name=row['name'],
            category=row.get('category', ''),
            tags=row.get('tags', []),
            instructions=row.get('instructions', ''),
            image_url=row.get('imageUrl', ''),
            ingredients=ingredient_names
        )

        cocktail_list.append(cocktail)

    return cocktail_list

def create_ingredients():
    """Creates and returns a list of unique ingredient names from the cocktail data."""
    # read cocktails from file
    data = load_cocktail_data()

    # prepare set for ingredients
    ingredients: set[Ingredient] = set()

    # create ingredient objects from data rows
    for row in data:
        # Skip invalid rows
        if not isinstance(row, dict) or 'ingredients' not in row:
            continue
            
        ingredients_data = row['ingredients']
        if not isinstance(ingredients_data, list):
            continue
            
        # create ingredient objects from cocktail ingredients
        for ingredient_data in ingredients_data:
            # Skip invalid ingredient data
            if not isinstance(ingredient_data, dict) or 'id' not in ingredient_data or 'name' not in ingredient_data:
                continue
                
            ingredient = Ingredient(
                id=ingredient_data['id'],
                name=ingredient_data['name'],
                description=ingredient_data.get('description', ''),
                alcohol=ingredient_data.get('alcoholic', False),
                type=ingredient_data.get('type', ''),
                image_url=ingredient_data.get('imageUrl', '')
            )
            ingredients.add(ingredient)
        

    return ingredients