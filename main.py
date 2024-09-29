# Import Module 
from bs4 import BeautifulSoup 
import requests 
from fractions import Fraction
import re
import json
import boto3
import os
import pandas as pd
from dotenv import load_dotenv

class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"
    
    def __init__(self, model_id="amazon.titan-embed-text-v2:0"):
        self.bedrock = boto3.client(service_name='bedrock-runtime')
        self.model_id = model_id
    def __call__(self, text, dimensions, normalize=True):
        """
        Returns Titan Embeddings
        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.
        Return:
            List[float]: Embedding
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })
        response = self.bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept=self.accept,
            contentType=self.content_type
        )
        response_body = json.loads(response.get('body').read())
        return response_body['embedding']
    
    def generate_embedding(self, text, dimensions, normalize=True):
        """
        Returns Titan Embeddings
        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.
        Return:
            List[float]: Embedding
        """
        return self(text, dimensions, normalize)

def convert_fractions_to_decimal(input_string):
    # Map for Unicode fractions
    unicode_fractions = {
        '¼': Fraction(1, 4),
        '½': Fraction(1, 2),
        '¾': Fraction(3, 4),
        '⅛': Fraction(1, 8),
        '⅜': Fraction(3, 8),
        '⅝': Fraction(5, 8),
        '⅞': Fraction(7, 8),
    }

    # Replace Unicode fractions with their decimal equivalents
    for unicode_frac, fraction in unicode_fractions.items():
        input_string = input_string.replace(unicode_frac, str(float(fraction)))

    # Regex pattern for unknown fractions at the start of the string (e.g., "1/2", "3/4")
    unknown_fraction_pattern = r'^\s*(\d+)\s*/\s*(\d+)'
    input_string = re.sub(unknown_fraction_pattern, 
                           lambda x: str(float(Fraction(int(x.group(1)), int(x.group(2))))), 
                           input_string)

    # Regex pattern for single whole numbers at the start of the string (e.g., "2")
    whole_number_pattern = r'^\s*(\d+)(?=\s|$)'
    input_string = re.sub(whole_number_pattern, 
                           lambda x: str(int(x.group(1))), 
                           input_string)

    # Regex pattern for mixed fractions at the start of the string (e.g., "1 1/2")
    mixed_fraction_pattern = r'^\s*(\d+)\s+(\d+)\s*/\s*(\d+)'
    input_string = re.sub(mixed_fraction_pattern, 
                           lambda x: str(float(Fraction(int(x.group(1)) * int(x.group(3)) + int(x.group(2)), int(x.group(3))))), 
                           input_string)

    return input_string

dimensions = 1024
normalize = True
    
titan_embeddings_v2 = TitanEmbeddings(model_id="amazon.titan-embed-text-v2:0")

input_text = "What are the different services that you offer?"
embedding = titan_embeddings_v2(input_text, dimensions, normalize)


# Load environment variables from .env file
load_dotenv()

# Define the required parameters from .env
APP_ID = os.getenv("APP_ID")
APP_KEY = os.getenv("APP_KEY")

# Function to get recipe data with pagination
def get_recipe_data(query, max_results=30, per_page=10):
    # Base API URL
    url = f'https://api.edamam.com/search?q={query}&app_id={APP_ID}&app_key={APP_KEY}'
    
    # Initialize list to store recipes
    all_recipes = []

    # Fetch results in pages
    for start in range(0, max_results, per_page):  # Pagination loop
        paginated_url = f'{url}&from={start}&to={start + per_page}'
        
        # Make the API call
        response = requests.get(paginated_url)
        print(f"GET {paginated_url}")
        print(f"Status Code: {response.status_code}")
        
        # Handle successful response
        if response.status_code == 200:
            data = response.json()
            recipes = data.get('hits', [])
            
            # Collecting recipes
            if recipes:
                for recipe in recipes:
                    recipe_info = recipe['recipe']
                    
                    # Store ingredients with their quantities and units
                    ingredients = []
                    for ingredient in recipe_info['ingredients']:
                        # Replace missing or placeholder values with 0
                        quantity = ingredient.get('quantity', 0) or 0
                        measure = ingredient.get('measure', '0')
                        if measure in [None, '<unit>']:
                            measure = '0'
                        food = ingredient.get('food', 'N/A')
                        ingredients.append(f"{quantity},{measure},{food}")
                    
                    # Create a dictionary for the recipe
                    recipe_dict = {
                        'Recipe Name': recipe_info['label'],
                        'URL': recipe_info['url'],
                        'Ingredients': ingredients,
                    }
                    # Append the recipe dictionary to the list
                    all_recipes.append(recipe_dict)
            else:
                print("No more recipes found.")
        else:
            print("Error fetching data:", response.text)

    return all_recipes

if __name__ == "__main__":
    dimensions = 1024
    normalize = True
    
    titan_embeddings_v2 = TitanEmbeddings(model_id="amazon.titan-embed-text-v2:0")
    
    # List of food items to search for
    food_list = ['chicken', 'beef', 'salmon', 'tofu']
    
    # Iterate through each food in the list
    for food in food_list:
        print(f"\nFetching recipes for: {food}\n")
        recipes = get_recipe_data(query=food, max_results=30,per_page=10)
        # Write the recipes to a text file in the specified format
        for recipe in recipes:
            text = ''
            text += f"URL: {recipe['URL']}\n"
            text += f"NAME: {recipe['Recipe Name']}\n"
            text += "INGREDIENTS:\n"
            for ingredient in recipe['Ingredients']:
                text += f"{ingredient}\n"
            embed = titan_embeddings_v2(text, dimensions, normalize)
            text += f"EMBEDDING: {embed}\n"
