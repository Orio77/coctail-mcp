import dotenv
import os
import json

from models.cocktail import Cocktail
from models.ingredient import Ingredient

dotenv.load_dotenv()

def load_cocktail_data():
    data_path = os.getenv("data_path")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
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
        ingredient_names = [ingredient['name'] for ingredient in row['ingredients']]

        cocktail = Cocktail(
            id=row['id'],
            name=row['name'],
            category=row['category'],
            tags=row['tags'],
            instructions=row['instructions'],
            image_url=row['imageUrl'],
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
        # create ingredient objects from cocktail ingredients
        for ingredient_data in row['ingredients']:
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