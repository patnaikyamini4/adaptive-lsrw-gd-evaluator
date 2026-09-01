from pymongo import MongoClient
from backend.config import MONGO_URI

client = MongoClient(MONGO_URI)

db = client["adaptive_lsrw_gd"]