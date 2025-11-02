from dataclasses import dataclass

@dataclass(frozen=True)
class Ingredient:
    """Represents an ingredient that can be used in cocktail recipes."""
    
    id: int  # Unique identifier for the ingredient
    name: str  # Display name of the ingredient (e.g., "Vodka", "Lime Juice")
    description: str  # Detailed description of the ingredient
    alcohol: bool  # Whether the ingredient contains alcohol
    type: str  # Category of ingredient (e.g., "Spirit", "Mixer", "Garnish")
    image_url: str  # URL to ingredient image