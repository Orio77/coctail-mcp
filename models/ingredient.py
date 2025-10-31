from dataclasses import dataclass

@dataclass
class Ingredient:
    id: int
    name: str
    description: str
    alcoholic: bool
    type: str
    image_url: str