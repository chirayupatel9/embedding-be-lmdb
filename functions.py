import os
import pandas as pd
from PIL import Image
import json
from datetime import datetime
import math

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


def create_sprites(image_path, output_folder, sprite_width, sprite_height):
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


def create_sprite_sheet(image_path, output_path, sprite_width, sprite_height, sheet_name="sprite_sheet.png"):
    """
    Creates a sprite sheet from individual images.

    Args:
        image_path (str): Path to the folder containing images.
        output_path (str): Path to save the sprite sheet.
        sprite_width (int): Width of each sprite.
        sprite_height (int): Height of each sprite.
        sheet_name (str): Name of the output sprite sheet image.

    Returns:
        dict: Metadata for the sprite sheet.
    """
    start = datetime.now()

    # Gather all image file paths
    image_filenames = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_filenames.append(os.path.join(root, file))

    if not image_filenames:
        raise ValueError("No valid image files found in the provided path.")

    # Calculate the number of rows and columns for the sprite sheet
    num_images = len(image_filenames)
    sheet_columns = math.ceil(math.sqrt(num_images))
    sheet_rows = math.ceil(num_images / sheet_columns)

    # Create the blank sprite sheet
    sheet_width = sheet_columns * sprite_width
    sheet_height = sheet_rows * sprite_height
    sprite_sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))

    # Place each image onto the sprite sheet
    sprite_metadata = {}
    for idx, image_file in enumerate(image_filenames):
        img = Image.open(image_file)

        # Convert to RGB if the image is in CMYK or other modes
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to fit the sprite dimensions
        img_resized = img.resize((sprite_width, sprite_height))

        # Calculate position on the sprite sheet
        col = idx % sheet_columns
        row = idx // sheet_columns
        x = col * sprite_width
        y = row * sprite_height

        # Paste the resized image onto the sprite sheet
        sprite_sheet.paste(img_resized, (x, y))

        # Add metadata for this sprite
        sprite_metadata[image_file] = {
            'x': x,
            'y': y,
            'width': sprite_width,
            'height': sprite_height,
            'sheet_name': sheet_name
        }

    # Save the sprite sheet
    os.makedirs(output_path, exist_ok=True)
    sprite_sheet_path = os.path.join(output_path, sheet_name)
    sprite_sheet.save(sprite_sheet_path)

    print(f"Sprite sheet created in {datetime.now() - start}")

    # Add overall metadata
    sprite_metadata["sprite_sheet"] = {
        'path': sprite_sheet_path,
        'columns': sheet_columns,
        'rows': sheet_rows,
        'width': sheet_width,
        'height': sheet_height
    }

    return sprite_metadata

# sprite_metadata = create_sprite_sheet(
#     image_path='static/train',
#     output_path='static/sprites',
#     sprite_width=64,
#     sprite_height=64
# )
#
# print("Sprite Sheet Metadata:\n", sprite_metadata)