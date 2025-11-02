from dataclasses import dataclass
from typing import List

@dataclass
class Cocktail:
    """Represents a cocktail recipe with ingredients and preparation instructions."""
    
    id: int  # Unique identifier for the cocktail
    name: str  # Display name of the cocktail
    category: str  # Category (e.g., "Martini", "Highball", "Shot")
    tags: List[str]  # Descriptive tags (e.g., ["sweet", "fruity", "strong"])
    instructions: str  # Step-by-step preparation instructions
    image_url: str  # URL to cocktail image
    ingredients: List[str]  # List of ingredients with measurements