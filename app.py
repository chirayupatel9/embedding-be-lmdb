from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import io
import json
from urllib.parse import unquote
from functions import *
from db_functions import get_all_documents as db_get_all_documents, get_document_by_field as db_get_document_by_field, delete_document as db_delete_document, get_all_images as db_get_all_images, get_image_with_details as db_get_image_with_details, upload_images_from_folder
from bson import ObjectId
import gridfs
from io import BytesIO

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory="output"), name="output")

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
            "spritePath": "/output/sprite_sheet.png",
            "metadataPath": "/output/metadata.json",
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
        metadata_path = "./output/metadata.json"
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
        
        # Read and return the metadata
        with open(metadata_path, "r") as file:
            json_data = json.load(file)
            
        return JSONResponse({
            "spritePath": "/output/sprite_sheet.png",
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
async def upload_images(folder_path: str = Form(...)):
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="Folder not found")
        response = upload_images_from_folder(folder_path)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
