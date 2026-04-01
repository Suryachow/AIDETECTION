from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from typing import List, Dict, Any, Optional
import logging
from core.config import settings

class LexicalDatabase:
    def __init__(self, connection_string: str = settings.MONGO_URI, 
                 database_name: str = settings.MONGO_DB):
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            self.collection = self.db["dictionary"]
            self.collection.create_index("word", unique=True)
        except ConnectionFailure:
            self.client = None

    def insert_multiple_entries(self, word_entries: List[Dict[str, Any]]):
        if not self.client: return
        for entry in word_entries:
            try:
                self.collection.insert_one(entry)
            except:
                continue

    def find_word(self, word: str) -> Optional[Dict[str, Any]]:
        if not self.client: return None
        return self.collection.find_one({"word": word.lower()}, {"_id": 0})
