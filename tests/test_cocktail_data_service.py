import sys
import os
import pytest
from unittest.mock import patch, mock_open
import json

# Add the parent directory to the path to import from the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.cocktail_data_service import create_cocktails, create_ingredients, load_cocktail_data
from models.cocktail import Cocktail
from models.ingredient import Ingredient


class TestLoadCocktailData:
    """Test cases for load_cocktail_data function"""
    
    def test_missing_data_path_env_var(self):
        """Test that ValueError is raised when data_path env var is not set"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="data_path environment variable is not set"):
                load_cocktail_data()
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised when data file doesn't exist"""
        with patch.dict(os.environ, {"data_path": "/nonexistent/path.json"}):
            with pytest.raises(FileNotFoundError, match="Data file not found at path"):
                load_cocktail_data()
    
    def test_invalid_json(self):
        """Test that ValueError is raised for invalid JSON"""
        mock_data = "invalid json content"
        with patch.dict(os.environ, {"data_path": "test.json"}):
            with patch("builtins.open", mock_open(read_data=mock_data)):
                with patch("os.path.exists", return_value=True):
                    with pytest.raises(ValueError, match="Invalid JSON in data file"):
                        load_cocktail_data()
    
    def test_non_list_data(self):
        """Test that ValueError is raised when data is not a list"""
        mock_data = json.dumps({"not": "a list"})
        with patch.dict(os.environ, {"data_path": "test.json"}):
            with patch("builtins.open", mock_open(read_data=mock_data)):
                with patch("os.path.exists", return_value=True):
                    with pytest.raises(ValueError, match="Data file must contain a list of cocktails"):
                        load_cocktail_data()
    
    def test_valid_data_loading(self):
        """Test successful data loading"""
        mock_data = json.dumps([{"id": 1, "name": "Test Cocktail"}])
        with patch.dict(os.environ, {"data_path": "test.json"}):
            with patch("builtins.open", mock_open(read_data=mock_data)):
                with patch("os.path.exists", return_value=True):
                    result = load_cocktail_data()
                    assert result == [{"id": 1, "name": "Test Cocktail"}]


class TestCreateCocktails:
    """Test cases for create_cocktails function"""
    
    def test_create_cocktails_empty_data(self):
        """Test create_cocktails with empty data"""
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=[]):
            cocktails = create_cocktails()
            assert cocktails == []
    
    def test_create_cocktails_valid_data(self):
        """Test create_cocktails with valid data"""
        mock_data = [
            {
                "id": 1,
                "name": "Mojito",
                "category": "Cocktail",
                "tags": ["refreshing", "mint"],
                "instructions": "Muddle mint leaves",
                "imageUrl": "http://example.com/mojito.jpg",
                "ingredients": [
                    {"id": 1, "name": "White Rum"},
                    {"id": 2, "name": "Mint"}
                ]
            }
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            cocktails = create_cocktails()
            assert len(cocktails) == 1
            assert isinstance(cocktails[0], Cocktail)
            assert cocktails[0].name == "Mojito"
            assert cocktails[0].id == 1
            assert cocktails[0].ingredients == ["White Rum", "Mint"]
    
    def test_create_cocktails_missing_required_fields(self):
        """Test create_cocktails skips rows with missing required fields"""
        mock_data = [
            {"id": 1, "name": "Valid Cocktail"},
            {"name": "Missing ID"},  # Missing id
            {"id": 2},  # Missing name
            "invalid_row",  # Not a dict
            {"id": 3, "name": "Another Valid"}
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            cocktails = create_cocktails()
            assert len(cocktails) == 2
            assert cocktails[0].name == "Valid Cocktail"
            assert cocktails[1].name == "Another Valid"
    
    def test_create_cocktails_with_missing_optional_fields(self):
        """Test create_cocktails handles missing optional fields"""
        mock_data = [
            {
                "id": 1,
                "name": "Simple Cocktail"
                # Missing all optional fields
            }
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            cocktails = create_cocktails()
            assert len(cocktails) == 1
            cocktail = cocktails[0]
            assert cocktail.category == ''
            assert cocktail.tags == []
            assert cocktail.instructions == ''
            assert cocktail.image_url == ''
            assert cocktail.ingredients == []


class TestCreateIngredients:
    """Test cases for create_ingredients function"""
    
    def test_create_ingredients_empty_data(self):
        """Test create_ingredients with empty data"""
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=[]):
            ingredients = create_ingredients()
            assert len(ingredients) == 0
    
    def test_create_ingredients_valid_data(self):
        """Test create_ingredients with valid data"""
        mock_data = [
            {
                "id": 1,
                "name": "Mojito",
                "ingredients": [
                    {
                        "id": 1,
                        "name": "White Rum",
                        "description": "Light rum",
                        "alcoholic": True,
                        "type": "Spirit",
                        "imageUrl": "http://example.com/rum.jpg"
                    },
                    {
                        "id": 2,
                        "name": "Mint",
                        "description": "Fresh mint leaves",
                        "alcoholic": False,
                        "type": "Herb"
                    }
                ]
            }
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            ingredients = create_ingredients()
            assert len(ingredients) == 2
            ingredient_names = {ing.name for ing in ingredients}
            assert "White Rum" in ingredient_names
            assert "Mint" in ingredient_names
    
    def test_create_ingredients_duplicate_removal(self):
        """Test that duplicate ingredients are removed"""
        mock_data = [
            {
                "id": 1,
                "name": "Cocktail 1",
                "ingredients": [
                    {"id": 1, "name": "Vodka"},
                    {"id": 2, "name": "Lime"}
                ]
            },
            {
                "id": 2,
                "name": "Cocktail 2",
                "ingredients": [
                    {"id": 1, "name": "Vodka"},  # Duplicate
                    {"id": 3, "name": "Cranberry"}
                ]
            }
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            ingredients = create_ingredients()
            assert len(ingredients) == 3  # Vodka should appear only once
            ingredient_names = {ing.name for ing in ingredients}
            assert ingredient_names == {"Vodka", "Lime", "Cranberry"}
    
    def test_create_ingredients_invalid_rows(self):
        """Test create_ingredients skips invalid rows and ingredients"""
        mock_data = [
            {
                "id": 1,
                "name": "Valid Cocktail",
                "ingredients": [
                    {"id": 1, "name": "Valid Ingredient"},
                    {"name": "Missing ID"},  # Missing id
                    {"id": 2},  # Missing name
                    "invalid_ingredient"  # Not a dict
                ]
            },
            {
                "id": 2,
                "name": "No Ingredients"
                # Missing ingredients field
            },
            "invalid_cocktail",  # Not a dict
            {
                "id": 3,
                "name": "Invalid Ingredients",
                "ingredients": "not_a_list"  # Not a list
            }
        ]
        
        with patch('services.cocktail_data_service.load_cocktail_data', return_value=mock_data):
            ingredients = create_ingredients()
            assert len(ingredients) == 1
            assert list(ingredients)[0].name == "Valid Ingredient"


def test_create_cocktails():
    """Simple test that calls create_cocktails and prints the result"""
    with patch('services.cocktail_data_service.load_cocktail_data', return_value=[
        {"id": 1, "name": "Test Cocktail", "ingredients": []}
    ]):
        cocktails = create_cocktails()
        print(f"Created {len(cocktails)} cocktails:")
        for cocktail in cocktails:
            print(f"- {cocktail.name}")


def test_create_ingredients():
    """Simple test that calls create_ingredients and prints the result"""
    with patch('services.cocktail_data_service.load_cocktail_data', return_value=[
        {"id": 1, "name": "Test Cocktail", "ingredients": [
            {"id": 1, "name": "Test Ingredient"}
        ]}
    ]):
        ingredients = create_ingredients()
        print(f"Created {len(ingredients)} unique ingredients:")
        for ingredient in ingredients:
            print(f"- {ingredient.name}")
