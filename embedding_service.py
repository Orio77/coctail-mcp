from models.cocktail import Cocktail
import ollama
import os
import dotenv

dotenv.load_dotenv()

def create_embeddings(cocktails: list[Cocktail]):
    model = os.getenv('embedding_model')
    dimensions = os.getenv('embedding_dimension')

    embeddings_map = {}

    for cocktail in cocktails:
        embedding_response = ollama.embed(model=model, input=repr(cocktail))
        # embedding_response.embeddings is a list of lists, take the first one
        embeddings_map[cocktail.id] = [float(val) for val in embedding_response.embeddings[0]]

    return embeddings_map