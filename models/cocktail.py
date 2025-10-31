from dataclasses import dataclass

@dataclass
class Cocktail:
    id: int
    name: str
    category: str
    tags: list[str]
    instructions: str
    image_url: str
    ingredients: list[str]