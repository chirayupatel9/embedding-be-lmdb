import os
import pandas as pd
from PIL import Image
import json
from datetime import datetime

def read_json_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def create_df(image_path):
    """Creates a DataFrame with file paths of images."""
    # Define the folder path
    folder_path = image_path

    # Initialize a list to store image filenames
    image_filenames = []

    # Walk through each directory and file in the folder path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Filter by image formats
                # Get the full path and add to the list
                file_path = os.path.join(root, file)
                image_filenames.append(file_path)

    # Create a pandas DataFrame from the list of filenames
    df_images = pd.DataFrame(image_filenames, columns=['File Path'])
    return df_images


def create_individual_sprites(image_path, output_folder, sprite_width, sprite_height):
    """Creates individual sprites for each image and saves them."""
    # Define the folder path
    start = datetime.now()
    folder_path = image_path
    # print(image_path, output_folder, sprite_width, sprite_height)
    # Initialize a list to store image filenames
    image_filenames = []

    # Walk through each directory and file in the folder path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # Filter by image formats
                # Get the full path and add to the list
                file_path = os.path.join(root, file)
                image_filenames.append(file_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store metadata for each individual sprite
    sprite_metadata = {}

    # Process each image to create individual sprites
    for idx, image_file in enumerate(image_filenames):
        # Load the image
        img = Image.open(image_file)

        # Convert to RGB if the image is in CMYK or other modes
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to the specified sprite size
        img_resized = img.resize((sprite_width, sprite_height))

        # Define the path to save the individual sprite
        sprite_filename = os.path.join(output_folder, f'sprite_{idx + 1}.png')
        img_resized.save(sprite_filename)

        # Store metadata for each sprite
        sprite_metadata[image_file] = {
            'sprite_path': sprite_filename,
            'width': sprite_width,
            'height': sprite_height
        }
    print(datetime.now()-start)
    return sprite_metadata

# Example usage:
# Create individual sprites and get metadata
# sprite_data = create_individual_sprites('static/train','static/sprites', 64, 64, )
# print("Individual Sprite Metadata:\n", sprite_data)

