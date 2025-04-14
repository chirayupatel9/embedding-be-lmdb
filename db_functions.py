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
print(get_all_images())
def get_image_with_details(image_id):
    """Retrieve an image along with its linked document"""
    try:
        # Fetch linked document
        document = collection.find_one({"image_id": str(image_id)})
        if document:
            document["_id"] = str(document["_id"])  # Convert ObjectId
            return document
        return None
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
def upload_images_from_folder(folder_path):
    """Upload all images in a folder and associate them with documents"""
    uploaded_files = []
    print(f"Uploading images from folder: {folder_path}")
    if not os.path.exists(folder_path):
        return {"error": "Folder not found"}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # print(f"Processing file: {file_path}")
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                # Upload image to GridFS
                # print(f"Uploading image: {filename}")
                img = f.read()
                # print(img)
                image_id = fs.put(img, filename=filename)
                # print(f"Image uploaded with ID: {image_id}")
                # Create document with image reference
                document_data = {
                    "filename": filename,
                    "image_id": str(image_id)
                }
                # print(f"Document data: {document_data}")
                result = collection.insert_one(document_data)
                # print(f"Document inserted with ID: {result.inserted_id}")
                uploaded_files.append({
                    "document_id": str(result.inserted_id),
                    "image_id": str(image_id),
                    "filename": filename
                })

    return {
        "message": "✅ Images uploaded successfully",
        "uploaded_files": uploaded_files
    }
# imgpath = '/home/cnp68/globus-data/3/409258704.png'
# if os.path.isfile(imgpath):
#     with open(imgpath, "rb") as f:
#         print(f.read())
# folder_path = "/home/cnp68/globus-data/3/"
# response = upload_images_from_folder(folder_path)
# print(response)67c77cc0325f1fb7a7aa6d90