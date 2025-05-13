from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests  # ✅ Add this line
import pymongo
import pdfplumber
import docx
import spacy
import shutil
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Get the OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"  
client = pymongo.MongoClient(MONGO_URI)
db = client["resume_matcher"]  
collection = db["resumes"]  
cover_letter_collection = db["cover_letters"]  # New Collection for Cover Letters


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Resume Uploader with NLP & Cover Letter Generator!"}

# ✅ Extract text from PDF
def extract_text_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# ✅ Extract text from DOCX
def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# ✅ Improved Skill Extraction (Handles Multi-word Skills)
def extract_skills(text):
    skill_keywords = [
        "TensorFlow", "TenserFlow", "Git", "GitHub", "Ms Excel", "Pandas", "Numpy", "Matplotlib", "NodeJS", "React", 
        "HTML", "CSS", "Java", "AI-ML", "JavaScript", "Python", "SQL", "C", "C++", "Machine Learning",
        "Deep Learning", "Data Science", "Docker", "AWS", "Azure", "FastAPI", "Django", "Flask",
        "Power BI", "Kubernetes"
    ]

    extracted_skills = set()
    text_lower = text.lower()
    
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            extracted_skills.add(skill)  

    return list(extracted_skills) if extracted_skills else ["No valid skills found."]

# ✅ Improved Experience Extraction
def extract_experience(text):
    experience_entries = []
    experience_patterns = [
        r"(?P<job_title>[\w\s]+?)\s*@\s*(?P<company>[\w\s]+?)\s*(?P<dates>\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4}(?:\s*-\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4})?\b)",
    ]

    responsibility_patterns = [
        r"(?P<responsibility>●.*)",  # Match bullet points (●)
        r"(?P<responsibility>- .*?)"  # Match dash bullets (- )
    ]

    extracted_experience = []
    current_experience = None

    lines = text.split("\n")

    for line in lines:
        # Match job title & company
        for pattern in experience_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # ✅ Save previous experience before starting a new one
                if current_experience:
                    extracted_experience.append(current_experience)

                current_experience = {
                    "job_title": match.group("job_title").strip(),
                    "company": match.group("company").strip(),
                    "dates": match.group("dates").strip(),
                    "responsibilities": []
                }
        
        # Match responsibilities
        if current_experience:
            for pattern in responsibility_patterns:
                match = re.search(pattern, line.strip())
                if match:
                    responsibility = match.group("responsibility").strip()
                    if responsibility and responsibility.lower() != "no details provided.":
                        current_experience["responsibilities"].append(responsibility)

    # ✅ Append the last extracted experience
    if current_experience:
        extracted_experience.append(current_experience)

    return extracted_experience if extracted_experience else ["No Experience Found"]
# ✅ Improved Project Extraction (Fixed 'tuple' issue)
def extract_projects(text):
    # ✅ Project Detection Patterns
    project_patterns = [
        r"\d+\)\s*([\w\s\(\)&-]+?)\s*(?:August|October|May|\b202\d\b)",  # Matches "1) Project Name August 2024"
        r"(?i)(?:developed|created|built|worked on)\s*[:-]?\s*([\w\s\(\)&-]+)",  # Matches "Developed XYZ"
        r"([\w\s&-]+)\s+(?:Website|Application|System|Tool|Software|Platform|Clone)",  # Matches "Amazon Clone"
    ]
    
    extracted_projects = set()

    for pattern in project_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            project_title = match.strip()
            if project_title and len(project_title) > 3:  # Avoid extracting short irrelevant words
                extracted_projects.add(project_title.title())  # Convert to proper case

    return list(extracted_projects) if extracted_projects else ["No projects found."]


# ✅ Improved Education Extraction
def extract_education(text):
    education_keywords = [
        "university", "college", "institute", "education",
        "bachelor", "master", "phd", "mba", "b.tech", "m.tech", "bsc", "msc"
    ]
    extracted_education = set()

    for line in text.split("\n"):
        if any(keyword.lower() in line.lower() for keyword in education_keywords):
            extracted_education.add(line.strip())

    return list(extracted_education) if extracted_education else ["No valid education found."]

# ✅ Extract structured resume data
def extract_keywords(text):
    doc = nlp(text)
    job_titles = set()

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "TITLE"]:  
            job_titles.add(ent.text)

    return {
        "skills": extract_skills(text),  
        "experience": extract_experience(text),  
        "education": extract_education(text),  
        "projects": extract_projects(text),
        "job_titles": list(job_titles) if job_titles else ["No job titles found."]
    }

# ✅ Extract ATS Keywords from Job Descriptions
def extract_ats_keywords(job_description):
    doc = nlp(job_description)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return list(set(keywords))

# ✅ Compute ATS Match Score (TF-IDF + OpenAI)
def compute_ats_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity_score = (vectors[0] @ vectors[1].T).toarray()[0][0]
    return round(similarity_score * 100, 2)  

# ✅ Define Request Models for API Input Validation
class JobDescriptionRequest(BaseModel):
    job_description: str

class CompareRequest(BaseModel):
    resume_text: str
    job_description: str

# ✅ Cover Letter Generation Model
class CoverLetterRequest(BaseModel):
    job_description: str
    resume_text: str    

# ✅ Upload Job Description & Extract ATS Keywords
@app.post("/job-description/")
async def upload_job_description(request: JobDescriptionRequest):
    try:
        ats_keywords = extract_ats_keywords(request.job_description)
        return {"status": "success", "ats_keywords": ats_keywords}
    except Exception as e:
        return {"status": "error", "message": f"Job description processing failed: {str(e)}"}

# ✅ Compare Resume with Job Description
@app.post("/compare/")
async def compare_resume_with_job(request: CompareRequest):
    try:
        print("Received Request Data:", request)  # ✅ Debugging
        ats_score = compute_ats_match_score(request.resume_text, request.job_description)
        return {"status": "success", "ats_score": ats_score}
    except Exception as e:
        return {"status": "error", "message": f"Comparison failed: {str(e)}"}

@app.post("/generate-cover-letter/")
def generate_cover_letter(request: CoverLetterRequest):
    try:
        ollama_api_url = "http://localhost:11434/api/generate"

        prompt = f"""
        Generate a professional cover letter for this job:
        
        Job Description:
        {request.job_description}
        
        Candidate Resume:
        {request.resume_text}
        
        Ensure it's structured, concise, and highlights relevant skills.
        """

        # ✅ Send JSON properly formatted
        response = requests.post(
            ollama_api_url,
            json={"model": "mistral", "prompt": prompt, "stream": False}  # Add "stream": False
        )

        # ✅ Check if response is valid JSON
        response_json = response.json()
        if "response" not in response_json:
            return {"status": "error", "message": "Invalid response format from Ollama."}

        cover_letter = response_json["response"]

        # ✅ Save to MongoDB
        cover_letter_collection.insert_one({
            "job_description": request.job_description,
            "resume_text": request.resume_text,
            "cover_letter": cover_letter
        })

        return {"status": "success", "cover_letter": cover_letter}
    
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}
    
    except ValueError as e:
        return {"status": "error", "message": f"JSON decoding failed: {str(e)}"}
    
    except Exception as e:
        return {"status": "error", "message": f"Cover letter generation failed: {str(e)}"}

# ✅ Existing Resume Processing & ATS Matching Functions (Unchanged)
# ✅ Extract text from PDF, DOCX, extract skills, experience, projects, etc.
# ✅ Upload job descriptions, extract ATS keywords, and compute ATS match scores

    
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # ✅ Save file
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Extract text
        if file.filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file_location)
        elif file.filename.endswith(".docx"):
            extracted_text = extract_text_from_docx(file_location)
        else:
            os.remove(file_location)  
            return {"status": "error", "message": "Unsupported file format. Use PDF or DOCX"}
        
        # ✅ Extract structured data
        resume_data = extract_keywords(extracted_text)

        # ✅ Store in MongoDB
        collection.insert_one({
            "filename": file.filename,
            "filepath": file_location,
            "text": extracted_text,
            "skills": resume_data["skills"],
            "experience": resume_data["experience"],
            "education": resume_data["education"],
            "projects": resume_data["projects"],
            "job_titles": resume_data["job_titles"]
        })

        # ✅ Ensure a clear response format
        return {
            "status": "success",
            "message": "File uploaded and processed successfully!",
            "filename": file.filename,
             "text": extracted_text,
            "skills": resume_data["skills"] if resume_data["skills"] else ["No skills found."],
            "experience": resume_data["experience"] if resume_data["experience"] else ["No experience found."],
            "education": resume_data["education"] if resume_data["education"] else ["No education found."],
            "projects": resume_data["projects"] if resume_data["projects"] else ["No projects found."],
            "job_titles": resume_data["job_titles"] if resume_data["job_titles"] else ["No job titles found."]
        }
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}
