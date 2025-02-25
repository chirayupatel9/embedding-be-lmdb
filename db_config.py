from pymongo import MongoClient
from bson import ObjectId
import gridfs
from dotenv import load_dotenv
from os.path import dirname,join
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)


# MongoDB connection details
MONGO_HOST = os.environ.get("MONGO_HOST", "localhost")
MONGO_PORT = int(os.environ.get("MONGO_PORT", 27017))
USERNAME = os.environ.get("USERNAME", "localuser")
PASSWORD = os.environ.get("PASSWORD", "localpassword")
AUTH_DB = os.environ.get("AUTH_DB", "admin")
DB_NAME = os.environ.get("DB_NAME", "testdb")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "dynamic_collection")

# Create MongoDB client
try:
    # mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.9
    client = MongoClient(f"mongodb://{USERNAME}:{PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{AUTH_DB}?authSource={AUTH_DB}")
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)  # Initialize GridFS for file storage
    collection = db[COLLECTION_NAME]
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print("❌ Error connecting to MongoDB:", e)


