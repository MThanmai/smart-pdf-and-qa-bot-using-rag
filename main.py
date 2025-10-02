from fastapi import FastAPI, status, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

app = FastAPI()

# CORS Configuration - Allow Streamlit Cloud
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://*.streamlit.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model once at startup
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

# Pydantic Models
class PdfRead(BaseModel):
    filename: str
    contents: str

class Chunking(BaseModel):
    text: str
    chunk_size: int = 500

class SemanticSearch(BaseModel):
    chunks: list[str]
    question: str
    top_k: int = 3

class AskGpt(BaseModel):
    question: str
    context: str
    api_key: str


# API Endpoints
@app.get("/")
def root():
    return {
        "message": "PDF Chatbot API is running!",
        "status": "healthy",
        "endpoints": {
            "pdf_upload": "/pdf-file",
            "chunk_text": "/chunk-text",
            "find_chunks": "/find-chunks",
            "ask_ai": "/ask-gpt"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/pdf-file", response_model=PdfRead, status_code=status.HTTP_200_OK)
async def pdf_file(file: UploadFile = File(...)):
    """Extract text from uploaded PDF"""
    try:
        pdf = await file.read()
        doc = fitz.open(stream=pdf, filetype="pdf")
        all_text = ""
        
        for page in doc:
            all_text += page.get_text()
        
        doc.close()
        
        return PdfRead(
            filename=file.filename,
            contents=all_text.strip()
        )
    except Exception as e:
        return {
            "filename": file.filename,
            "contents": f"Error reading PDF: {str(e)}"
        }

@app.post("/chunk-text")
def chunk_text(data: Chunking):
    """Split text into chunks of specified size"""
    try:
        words = data.text.split()
        chunks = [
            ' '.join(words[i:i+data.chunk_size]) 
            for i in range(0, len(words), data.chunk_size)
        ]
        return {
            'chunks': chunks,
            'total_chunks': len(chunks)
        }
    except Exception as e:
        return {
            'chunks': [],
            'error': str(e)
        }

@app.post("/find-chunks")
def find_chunks(data: SemanticSearch):
    """Find most relevant chunks using semantic similarity"""
    try:
        # Encode chunks and question
        chunk_embeddings = model.encode(data.chunks)
        question_embedding = model.encode([data.question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(question_embedding, chunk_embeddings).flatten()
        
        # Get top K most similar chunks
        top_indices = similarities.argsort()[-data.top_k:][::-1]
        top_chunks = [data.chunks[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        
        return {
            "top_chunks": top_chunks,
            "similarity_scores": top_scores
        }
    except Exception as e:
        return {
            "top_chunks": [],
            "error": str(e)
        }

@app.post("/ask-gpt", status_code=status.HTTP_200_OK)
def ask_gpt(data: AskGpt):
    """Generate answer using Google Gemini (Free API)"""
    
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context in a concise and accurate way.

Context:
{data.context}

Question:
{data.question}

Answer:"""
    
    try:
        # Configure Gemini with API key
        genai.configure(api_key=data.api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        answer = response.text
        
        return {
            'answer': answer,
            'model': 'gemini-1.5-flash'
        }
    
    except Exception as e:
        return {
            'answer': f"Error generating answer: {str(e)}. Please check your API key at ai.google.dev",
            'error': True
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
