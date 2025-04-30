from pymongo import MongoClient
from bson import ObjectId
import gridfs
from dotenv import load_dotenv
from os.path import dirname, join
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv()

# MongoDB connection details
MONGO_HOST = os.environ.get("MONGO_HOST", "m3kube.urcf.drexel.edu")
MONGO_PORT = int(os.environ.get("MONGO_PORT", 27017))
USERNAME = os.environ.get("MONGO_USERNAME", "mongouser")
PASSWORD = os.environ.get("MONGO_PASSWORD", "password123")
AUTH_DB = os.environ.get("MONGO_AUTH_DB", "admin")
DB_NAME = os.environ.get("MONGO_DB_NAME", "2022_materials_project_3")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION", "dynamic_collection")

# Create MongoDB client
try:
    # Construct MongoDB URI
    mongo_uri = f"mongodb://{USERNAME}:{PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{AUTH_DB}?authSource={AUTH_DB}&retryWrites=true&w=majority"
    
    # Create client with connection timeout
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.server_info()
    
    # Initialize database and collections
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    collection = db[COLLECTION_NAME]
    
    print("✅ Connected to MongoDB successfully!")
    print(f"🔍 Using database: {DB_NAME}")
    print(f"🔍 Using collection: {COLLECTION_NAME}")

except Exception as e:
    print(f"❌ Error connecting to MongoDB: {str(e)}")
    raise



