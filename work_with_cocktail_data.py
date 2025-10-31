import dotenv
import os
import json

from models.cocktail import Cocktail

dotenv.load_dotenv()

# function to create cocktail objects from data file
def create_cocktails():

    # read cocktails from file
    data_path = os.getenv("data_path")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

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