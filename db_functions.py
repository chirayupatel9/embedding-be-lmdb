from pymongo import MongoClient
from bson import ObjectId
import gridfs
import json
from dotenv import load_dotenv
from os.path import dirname,join
import os
from db_config import *

# Convert MongoDB ObjectId to string
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

# CRUD Functions with Image Linking
def create_document_with_image(document_data, filename, file_data):
    """Insert a new document with an image"""
    image_id = fs.put(file_data, filename=filename)  # Save image in GridFS
    document_data["image_id"] = str(image_id)  # Store image reference in document

    result = collection.insert_one(document_data)
    return {
        "message": "✅ Document inserted with image",
        "document_id": str(result.inserted_id),
        "image_id": str(image_id)
    }


def get_all_documents():
    """Fetch all documents with image URLs"""
    documents = list(collection.find())  # Convert cursor to list

    for doc in documents:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        if "image_id" in doc:
            doc["image_url"] = f"/get-image/{doc['image_id']}"  # Add image URL reference

    return documents  # Return list directly


def get_document_by_field(field, value):
    """Find a document dynamically by any field, including linked images"""
    document = collection.find_one({field: value})
    if document:
        if "image_id" in document:
            document["image_url"] = f"/get-image/{document['image_id']}"  # Add image URL reference
        return json.loads(json.dumps(document, cls=JSONEncoder))
    return None


def delete_document(field, value):
    """Delete a document and its linked image"""
    document = collection.find_one({field: value})
    if not document:
        return None

    # Delete the associated image from GridFS
    if "image_id" in document:
        try:
            fs.delete(ObjectId(document["image_id"]))
        except Exception:
            pass

    # Delete the document itself
    result = collection.delete_one({field: value})
    if result.deleted_count > 0:
        return {"message": "✅ Document and associated image deleted"}
    return None


# Image Storage Functions

def get_image(image_id):
    """Retrieve an image from MongoDB using GridFS"""
    try:
        file = fs.get(ObjectId(image_id))
        return file.read()
    except Exception:
        return None

# 🚀 New Image Handling Functions

def get_all_images():
    """Fetch metadata of all stored images"""
    images = []
    for file in fs.find():
        images.append({
            "image_id": str(file._id),
            "filename": file.filename,
            "image_url": f"/get-image/{file._id}"
        })

    return images
# print(get_all_images())
def get_image_with_details(image_id):
    """Retrieve an image along with its linked document
    
    Args:
        image_id (str): ID of the image to retrieve
        
    Returns:
        dict: Document with image details or None if not found
    """
    try:
        # First check if image exists
        try:
            file = fs.get(ObjectId(image_id))
            file_metadata = {
                "filename": file.filename,
                "content_type": file.content_type if hasattr(file, "content_type") else "image/jpeg",
                "upload_date": file.upload_date if hasattr(file, "upload_date") else None,
                "length": file.length if hasattr(file, "length") else None
            }
        except:
            return None
            
        # Fetch linked document
        document = collection.find_one({"image_id": str(image_id)})
        if document:
            document["_id"] = str(document["_id"])  # Convert ObjectId
            # Add file metadata info
            document["gridfs_metadata"] = file_metadata
            return document
            
        # If no document found, return basic file info
        return file_metadata
    except Exception as e:
        print(f"Error in get_image_with_details: {str(e)}")
        return None


def get_all_images_with_details():
    """Retrieve all images with their associated document details"""
    images_with_details = []

    for file in fs.find():
        image_id = str(file._id)
        document = collection.find_one({"image_id": image_id})
        if document:
            document["_id"] = str(document["_id"])
            images_with_details.append({
                "image_id": image_id,
                "filename": file.filename,
                "image_url": f"/get-image/{image_id}",
                "document_details": document
            })

    return images_with_details




# 🚀 New Function to Upload All Images from a Folder
def upload_images_from_folder(folder_path, metadata_folder=None, exact_match=False):
    """Upload all images in a folder and associate them with documents
    
    Args:
        folder_path (str): Path to folder containing images
        metadata_folder (str, optional): Path to folder containing JSON metadata files
        exact_match (bool): If True, requires exact filename match (excluding extension)
    """
    uploaded_files = []
    print(f"Uploading images from folder: {folder_path}")
    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}
    
    # Check if metadata folder exists (if provided)
    if metadata_folder and not os.path.exists(metadata_folder):
        return {"error": "Metadata folder not found"}
    
    # Load metadata if provided
    metadata_dict = {}
    if metadata_folder:
        metadata_files = [f for f in os.listdir(metadata_folder) if f.lower().endswith('.json')]
        metadata_dict = {os.path.splitext(meta)[0]: os.path.join(metadata_folder, meta) for meta in metadata_files}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Skip if not a file or not an image
        if not os.path.isfile(file_path) or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        try:
            with open(file_path, "rb") as f:
                # Upload image to GridFS
                img = f.read()
                image_id = fs.put(img, filename=filename)
                
                # Base document data
                document_data = {
                    "filename": filename,
                    "image_id": str(image_id)
                }
                
                # Try to find and add metadata if metadata_folder was provided
                if metadata_folder:
                    image_base_name = os.path.splitext(filename)[0]
                    metadata_file_path = None
                    
                    # Exact matching
                    if exact_match:
                        if image_base_name in metadata_dict:
                            metadata_file_path = metadata_dict[image_base_name]
                    # Flexible matching
                    else:
                        # Direct match first
                        if image_base_name in metadata_dict:
                            metadata_file_path = metadata_dict[image_base_name]
                        else:
                            # Try partial matches
                            potential_matches = [meta for meta in metadata_dict.keys() 
                                               if image_base_name in meta or meta in image_base_name]
                            if potential_matches:
                                metadata_file_path = metadata_dict[potential_matches[0]]
                    
                    # If metadata found, load and merge it with document_data
                    if metadata_file_path:
                        with open(metadata_file_path, 'r') as meta_file:
                            try:
                                metadata = json.load(meta_file)
                                if isinstance(metadata, dict):
                                    # Record the metadata filename
                                    metadata['metadata_filename'] = os.path.basename(metadata_file_path)
                                    metadata['image_filename'] = filename
                                    metadata['image_id'] = str(image_id)
                                    document_data = metadata  # Replace with full metadata
                                else:
                                    # If metadata is not a dict, just add it as a field
                                    document_data['metadata'] = metadata
                            except json.JSONDecodeError:
                                # If JSON is invalid, continue without metadata
                                document_data['metadata_error'] = f"Invalid JSON in {metadata_file_path}"
                
                # Insert document with metadata (if found) or basic info
                result = collection.insert_one(document_data)
                
                uploaded_files.append({
                    "document_id": str(result.inserted_id),
                    "image_id": str(image_id),
                    "filename": filename,
                    "has_metadata": 'metadata_filename' in document_data
                })

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    return {
        "message": f"✅ Images uploaded successfully ({len(uploaded_files)} files)",
        "uploaded_files": uploaded_files
    }
# imgpath = '/home/cnp68/globus-data/3/409258704.png'
# if os.path.isfile(imgpath):
#     with open(imgpath, "rb") as f:
#         print(f.read())
# folder_path = "/home/cnp68/globus-data/3/"
# response = upload_images_from_folder(folder_path)
# print(response)67c77cc0325f1fb7a7aa6d90

def get_metadata_image_relationships(filter_by=None):
    """
    Get relationships between metadata and images based on their filenames.
    
    Args:
        filter_by (dict, optional): MongoDB filter to apply
        
    Returns:
        List of documents showing relationships between images and metadata
    """
    query = filter_by if filter_by else {}
    documents = list(collection.find(query))
    
    relationships = []
    for doc in documents:
        relationship = {
            "document_id": str(doc["_id"]),
            "image_id": doc.get("image_id"),
            "image_filename": doc.get("image_filename", "unknown"),
            "metadata_filename": doc.get("metadata_filename", "unknown"),
            "has_valid_relationship": bool(doc.get("image_id") and doc.get("metadata_filename"))
        }
        
        # Add other relevant fields if available
        for field in ["label", "category", "description"]:
            if field in doc:
                relationship[field] = doc[field]
                
        relationships.append(relationship)
        
    return relationships