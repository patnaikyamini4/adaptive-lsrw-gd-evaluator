from pymongo import MongoClient
from backend.config import MONGO_URI

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=30000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000
)

db = client["adaptive_lsrw_gd"]