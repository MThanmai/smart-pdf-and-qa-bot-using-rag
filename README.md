MindVault - Smart PDF Q&A Bot

MindVault is an intelligent document chat system that lets you ask questions about your documents using natural language. Built with Retrieval-Augmented Generation (RAG) technology, it combines semantic search with AI to provide accurate, context-aware answers.

🎯 What It Does
Upload your documents (PDF, Word, TXT, Markdown) and chat with them naturally. MindVault finds relevant information and generates accurate answers with source citations.

🛠️ Technologies Used

Frontend: Streamlit
Vector Database: ChromaDB (HNSW algorithm)
Embeddings: all-MiniLM-L6-v2 (sentence-transformers)
LLM: Ollama (Local - llama3.2, mistral, phi3, or any Ollama model)
Document Processing: PyPDF2, python-docx
Search Strategy: Hybrid (Semantic + Keyword with 70/30 weighting)

🔑 Key Features

Semantic search using vector embeddings (384-dimensional)
Smart text chunking with 500-char chunks and 50-char overlap
Hybrid retrieval combining semantic and keyword search
Persistent local storage
Source citation with similarity scores
100% free - no API costs required


Built with: Python 3.9+ | ChromaDB | Sentence Transformers | Streamlit
