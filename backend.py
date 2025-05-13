# backend/main.py
from fastapi import FastAPI, File, UploadFile
from PyPDF2 import PdfReader
from docx import Document
import openai
import os

app = FastAPI()



def extract_text(file: UploadFile):
    """Extract text from uploaded PDF or DOCX"""
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text(file)
    
    if not text:
        return {"error": "Could not extract text"}

    # Basic ATS keyword match using OpenAI embeddings
    ats_keywords = ["Python", "JavaScript", "Machine Learning", "React", "AWS"]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"Extract key skills and match against: {ats_keywords}"},
                  {"role": "user", "content": text}]
    )

    return {
        "filename": file.filename,
        "extracted_text": text[:500],  # Limit output for display
        "AI_analysis": response["choices"][0]["message"]["content"]
    }
