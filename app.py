from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import json
from urllib.parse import unquote
from functions import *
from db_functions import get_all_documents as db_get_all_documents, get_document_by_field as db_get_document_by_field, delete_document as db_delete_document, get_all_images as db_get_all_images, get_image_with_details as db_get_image_with_details, upload_images_from_folder, get_metadata_image_relationships
from bson import ObjectId
import gridfs
from io import BytesIO
import numpy as np
import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173","0.0.0.0:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/output", StaticFiles(directory="api/output"), name="/api/output")

@app.get("/api/generate-sprite-sheet")
async def generate_sprite_sheet(method: str = "tsne"):
    """
    Generate a sprite sheet and metadata from MongoDB images.
    Args:
        method (str): Dimensionality reduction method ("tsne" or "umap")
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output paths
        output_sprite = os.path.join(output_dir, "sprite_sheet.png")
        output_json = os.path.join(output_dir, "metadata.json")
        
        # Generate sprite sheet from MongoDB
        result = create_sprite_sheet_from_mongodb(
            output_sprite=output_sprite,
            output_json=output_json,
            reduction_method=method
        )
        
        return JSONResponse({
            "spritePath": "/api/output/sprite_sheet.png",
            "metadataPath": "/api/output/metadata.json",
            "numImages": result["num_images"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/embeddings")
async def get_embeddings(method: str = "tsne"):
    """
    Get the sprite sheet and metadata. If they don't exist, generate them first.
    Args:
        method (str): Dimensionality reduction method ("tsne" or "umap")
    """
    try:
        # Check if metadata file exists
        metadata_path = f"./output/{method}_metadata.json"
        sprite_path = "./output/sprite_sheet.png"
        
        if not os.path.exists(metadata_path) or not os.path.exists(sprite_path):
            # Generate sprite sheet and metadata if they don't exist
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            result = create_sprite_sheet_from_mongodb(
                output_sprite=sprite_path,
                output_json=metadata_path,
                reduction_method=method
            )
            
            if not result:
                raise HTTPException(status_code=500, detail="Failed to generate sprite sheet")
        
        # Read the metadata
        with open(metadata_path, "r") as file:
            json_data = json.load(file)
            
        # Calculate sprite sheet dimensions
        num_images = len(json_data)
        sprite_dim = int(np.ceil(np.sqrt(num_images)))
        sprite_width = 32  # Each sprite is 32x32 pixels
        sprite_height = 32
        
        return JSONResponse({
            "spritePath": {
                "columns": sprite_dim,
                "rows": sprite_dim,
                "width": sprite_dim * sprite_width,
                "height": sprite_dim * sprite_height,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
                "url": "/output/sprite_sheet.png"
            },
            "itemsPath": json_data
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Failed to generate or find metadata file")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid metadata file format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-image/{image_id}")
async def get_image(image_id: str):
    try:
        file = fs.get(ObjectId(image_id))
        image_bytes = file.read()
        return StreamingResponse(BytesIO(image_bytes), media_type="image/jpeg")
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/read")
async def get_all_documents():
    try:
        documents = db_get_all_documents()
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/read/{field}/{value}")
async def get_document_by_field(field: str, value: str):
    try:
        document = db_get_document_by_field(field, value)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/delete/{field}/{value}")
async def delete_document(field: str, value: str):
    try:
        response = db_delete_document(field, value)
        if not response:
            raise HTTPException(status_code=404, detail="Document not found")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-all-images")
async def get_all_images():
    try:
        images = db_get_all_images()
        if not images:
            raise HTTPException(status_code=404, detail="No images found")
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-image-details/{image_id}")
async def get_image_with_details(image_id: str):
    try:
        document_data = db_get_image_with_details(image_id)
        if not document_data:
            raise HTTPException(status_code=404, detail="Image or document not found")
        return {
            "image_id": image_id,
            "document_details": document_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-images")
async def upload_images(
    folder_path: str = Form(...),
    metadata_folder: str = Form(None),
    exact_match: bool = Form(False)
):
    """
    Upload images from a folder and optionally associate with metadata.
    
    Args:
        folder_path (str): Path to the folder containing images
        metadata_folder (str, optional): Path to folder containing metadata files
        exact_match (bool, optional): If True, requires exact filename match for metadata
    """
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="Folder not found")
        
        if metadata_folder and not os.path.exists(metadata_folder):
            raise HTTPException(status_code=400, detail="Metadata folder not found")
            
        response = upload_images_from_folder(folder_path, metadata_folder, exact_match)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-images-with-metadata")
async def upload_images_with_metadata(
    images_folder: str = Form(...),
    metadata_folder: str = Form(...),
    exact_match: bool = Form(False)
):
    """
    Upload images and their corresponding metadata from two separate folders.
    Args:
        images_folder (str): Path to the folder containing images
        metadata_folder (str): Path to the folder containing JSON metadata files
        exact_match (bool, optional): If True, requires exact filename match (excluding extension).
                                     If False, uses more flexible matching (default)
    """
    try:
        if not os.path.exists(images_folder) or not os.path.exists(metadata_folder):
            raise HTTPException(status_code=400, detail="One or both folders do not exist")

        # Get list of image files and metadata files
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        metadata_files = [f for f in os.listdir(metadata_folder) if f.lower().endswith('.json')]

        if not image_files:
            raise HTTPException(status_code=400, detail="No image files found in the images folder")
        if not metadata_files:
            raise HTTPException(status_code=400, detail="No metadata files found in the metadata folder")

        uploaded_count = 0
        errors = []
        
        # Create dictionaries for easier lookup
        # For images: strip extension and use as key
        image_dict = {os.path.splitext(img)[0]: img for img in image_files}
        # For metadata: strip extension and use as key
        metadata_dict = {os.path.splitext(meta)[0]: meta for meta in metadata_files}
        
        # Process each image and find its corresponding metadata
        for image_base_name, image_file in image_dict.items():
            try:
                metadata_file = None
                metadata = None
                
                # Exact matching - requires exact filename match (excluding extension)
                if exact_match:
                    if image_base_name in metadata_dict:
                        metadata_file = metadata_dict[image_base_name]
                # Flexible matching - try various approaches
                else:
                    # First try direct match
                    if image_base_name in metadata_dict:
                        metadata_file = metadata_dict[image_base_name]
                    else:
                        # Try to find metadata file that contains the image name
                        potential_matches = [meta for meta_base in metadata_dict.keys() 
                                            if image_base_name in meta_base or meta_base in image_base_name]
                        if potential_matches:
                            # Use the first potential match
                            metadata_file = metadata_dict[potential_matches[0]]
                
                if not metadata_file:
                    errors.append(f"No metadata found for {image_file}")
                    continue
                
                # Read the image
                image_path = os.path.join(images_folder, image_file)
                with open(image_path, 'rb') as img_file:
                    image_data = img_file.read()
                
                # Read the metadata
                metadata_path = os.path.join(metadata_folder, metadata_file)
                with open(metadata_path, 'r') as meta_file:
                    metadata = json.load(meta_file)
                
                # Store image in GridFS
                image_id = str(fs.put(image_data, filename=image_file))
                
                # Add image_id and filename to metadata and store in MongoDB
                metadata['image_id'] = image_id
                metadata['image_filename'] = image_file
                metadata['metadata_filename'] = metadata_file
                collection.insert_one(metadata)
                
                uploaded_count += 1
                
            except Exception as e:
                errors.append(f"Error processing {image_file}: {str(e)}")
                continue
                
        return {
            "message": f"Successfully uploaded {uploaded_count} images with metadata",
            "errors": errors if errors else None
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/relationships")
async def get_relationships(field: str = None, value: str = None):
    """
    Get relationships between images and metadata.
    
    Args:
        field (str, optional): Field to filter by
        value (str, optional): Value to filter by
    
    Returns:
        List of documents showing relationships between images and metadata
    """
    try:
        # Apply filter if provided
        filter_by = {field: value} if field and value else None
        relationships = get_metadata_image_relationships(filter_by)
        
        if not relationships:
            raise HTTPException(status_code=404, detail="No relationships found")
            
        return {
            "count": len(relationships),
            "relationships": relationships
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image-with-metadata/{image_id}")
async def get_image_with_metadata(image_id: str):
    """
    Get both image data (as base64) and associated metadata in a single response.
    
    Args:
        image_id (str): ID of the image to retrieve
    """
    try:
        # Get image from GridFS
        try:
            file = fs.get(ObjectId(image_id))
            image_bytes = file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        except gridfs.errors.NoFile:
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Get associated metadata document
        document = collection.find_one({"image_id": image_id})
        if not document:
            # Return image with minimal metadata if no document found
            return {
                "image_id": image_id,
                "filename": getattr(file, "filename", "unknown"),
                "image_data": image_base64,
                "content_type": "image/jpeg",
                "has_metadata": False
            }
            
        # Convert ObjectId to string
        document["_id"] = str(document["_id"])
        
        # Return combined response
        return {
            "image_id": image_id,
            "filename": getattr(file, "filename", "unknown"),
            "image_data": image_base64,
            "content_type": "image/jpeg",
            "has_metadata": True,
            "metadata": document
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
  
  
@app.get("/api/dimensionality-reduction/{method}")
async def dimensionality_reduction(method: str):
    """
    Perform dimensionality reduction using either UMAP or t-SNE.
    Args:
        method (str): Dimensionality reduction method ("tsne" or "umap")
    Returns:
        JSONResponse: Contains the reduced coordinates and metadata
    """
    try:
        if method.lower() not in ["tsne", "umap"]:
            raise HTTPException(status_code=400, detail="Method must be either 'tsne' or 'umap'")
        
        # Perform dimensionality reduction
        if method.lower() == "tsne":
            metadata_path = f"./output/{method}_metadata.json"
        else:  # umap
            metadata_path = f"./output/{method}_metadata.json"
        
        # Prepare response
        with open(metadata_path, "r") as file:
            json_data = json.load(file)
            
        # Calculate sprite sheet dimensions
        num_images = len(json_data)
        sprite_dim = int(np.ceil(np.sqrt(num_images)))
        sprite_width = 32  # Each sprite is 32x32 pixels
        sprite_height = 32
        
        return JSONResponse({
            "spritePath": {
                "columns": sprite_dim,
                "rows": sprite_dim,
                "width": sprite_dim * sprite_width,
                "height": sprite_dim * sprite_height,
                "sprite_width": sprite_width,
                "sprite_height": sprite_height,
                "url": "/output/sprite_sheet.png"
            },
            "itemsPath": json_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))