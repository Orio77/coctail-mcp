from dataclasses import dataclass

@dataclass(frozen=True)
class Ingredient:
    id: int
    name: str
    description: str
    alcohol: bool
    type: str
    image_url: str