import os
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List
from bson import ObjectId
from datetime import datetime
from dependencies import verify_firebase_token, get_encrypted_notes_connection

router = APIRouter(prefix="/notepad", tags=["Notepad Engine"])

class CategoryCreate(BaseModel):
    name: str

class NoteCreate(BaseModel):
    category_id: str
    title: str
    content: str

@router.post("/categories")
async def create_category(data: CategoryCreate, decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    db = get_encrypted_notes_connection(user_id)
    
    existing = db.categories.find_one({"user_id": user_id, "name": data.name.strip()})
    if existing:
        raise HTTPException(status_code=400, detail="Category storage mismatch.")
        
    try:
        result = db.categories.insert_one({
            "user_id": user_id,
            "name": data.name.strip()
        })
        return {"status": "success", "category_id": str(result.inserted_id), "name": data.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Category storage mismatch.")

@router.get("/categories")
async def get_user_categories(decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    db = get_encrypted_notes_connection(user_id)
    
    cursor = db.categories.find({"user_id": user_id})
    return [{"id": str(doc["_id"]), "name": doc["name"]} for doc in cursor]

@router.delete("/categories/{category_id}")
async def delete_category(category_id: str, decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    try:
        db = get_encrypted_notes_connection(user_id)
        
        result = db.categories.delete_one({"_id": ObjectId(category_id), "user_id": user_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Category not found.")
            
        db.notes.delete_many({"category_id": category_id, "user_id": user_id})
        
        return {"status": "success", "message": "Category context destroyed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/notes")
async def create_note(data: NoteCreate, decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    try:
        db = get_encrypted_notes_connection(user_id)
        
        category = db.categories.find_one({"_id": ObjectId(data.category_id), "user_id": user_id})
        if not category:
            raise HTTPException(status_code=403, detail="Scope mismatch.")
            
        result = db.notes.insert_one({
            "user_id": user_id,
            "category_id": data.category_id,
            "title": data.title.strip(),
            "content": data.content.strip(),
            "updated_at": datetime.utcnow().isoformat()
        })
        return {"status": "success", "note_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/notes/{category_id}")
async def get_notes_by_category(category_id: str, decoded_token: dict = Depends(verify_firebase_token)):
    user_id = decoded_token["uid"].strip().lower()
    db = get_encrypted_notes_connection(user_id)
    
    category = db.categories.find_one({"_id": ObjectId(category_id), "user_id": user_id})
    if not category:
        return []
        
    cursor = db.notes.find({"category_id": category_id, "user_id": user_id}).sort("_id", -1)
    return [{"id": str(doc["_id"]), "title": doc["title"], "content": doc["content"], "updated_at": doc["updated_at"]} for doc in cursor]