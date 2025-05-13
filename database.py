# backend/database.py
from pymongo import MongoClient

client = MongoClient("mongodb+srv://your_mongo_uri")
db = client.resume_analyzer

def save_resume(data):
    db.resumes.insert_one(data)
