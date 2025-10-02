import streamlit as st 
import requests 
import os

# Backend URL - Update this after deploying to Render
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="Smart PDF QA Bot", 
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📄 Smart PDF QA Bot")
st.markdown("### Upload any PDF and ask questions - powered by AI 🚀")

# Sidebar
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    **Step 1:** Upload a PDF file
    
    **Step 2:** Get a free Gemini API key:
    - Visit [ai.google.dev](https://ai.google.dev)
    - Click "Get API Key"
    - Sign in with Google
    - Create a new API key
    
    **Step 3:** Enter the API key below
    
    **Step 4:** Click "Process PDF"
    
    **Step 5:** Ask questions!
    """)
    
    st.markdown("---")
    st.markdown("**⚡ Tech Stack:**")
    st.markdown("- 🔧 FastAPI Backend")
    st.markdown("- 🎨 Streamlit Frontend")
    st.markdown("- 🤖 Google Gemini (Free)")
    st.markdown("- 🧠 Sentence Transformers")
    st.markdown("- 📊 Semantic Search")
    
    st.markdown("---")
    st.markdown("**💡 Tips:**")
    st.markdown("- Works best with text-based PDFs")
    st.markdown("- First query may take 30-60s if backend is sleeping")
    st.markdown("- Ask specific questions for better answers")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file:",
        type=["pdf"],
        help="Upload any PDF document to analyze"
    )

with col2:
    st.subheader("🔑 API Configuration")
    api_key = st.text_input(
        "Gemini API Key:",
        type="password",
        help="Get free key at ai.google.dev",
        placeholder="Enter your API key here"
    )

# Process PDF button
if uploaded_file and api_key:
    st.session_state.api_key = api_key
    
    st.markdown("---")
    
    if st.button("🔄 Process PDF", use_container_width=True):
        with st.spinner("📖 Reading and analyzing PDF... This may take a moment."):
            try:
                # Step 1: Upload PDF to backend
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                
                response = requests.post(
                    f"{backend_url}/pdf-file",
                    files=files,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['contents']
                    
                    # Check if content was extracted
                    if not content or len(content.strip()) < 50:
                        st.error("⚠️ Could not extract text from PDF. Make sure it's not a scanned image.")
                    else:
                        st.session_state.content = content
                        
                        # Step 2: Chunk the text
                        response2 = requests.post(
                            f"{backend_url}/chunk-text",
                            json={'text': content, 'chunk_size': 500},
                            timeout=60
                        )
                        
                        if response2.status_code == 200:
                            result2 = response2.json()
                            chunks = result2['chunks']
                            st.session_state.chunks = chunks
                            
                            # Success message
                            st.success(f"✅ PDF processed successfully!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("📄 Pages", "Processed")
                            with col_b:
                                st.metric("📊 Total Words", f"{len(content.split()):,}")
                            with col_c:
                                st.metric("🧩 Chunks Created", len(chunks))
                            
                            st.info("👇 Now you can ask questions about this PDF below!")
                        else:
                            st.error(f"❌ Error chunking text: {response2.status_code}")
                else:
                    st.error(f"❌ Error reading PDF: {response.status_code}")
                    st.write(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to backend server!")
                st.info(f"Make sure backend is running at: {backend_url}")
                st.code(f"Backend URL: {backend_url}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try with a smaller PDF.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Question Answering Section
if "chunks" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask Questions About Your PDF")
    
    # Sample questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is the main topic of this document?
        - Summarize the key points
        - What are the conclusions?
        - Who are the authors?
        - What methodology was used?
        """)
    
    user_question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is the main topic of this document?",
        help="Ask anything about the PDF content"
    )
    
    if user_question:
        with st.spinner("🔍 Finding the best answer..."):
            try:
                # Step 3: Find relevant chunks
                response3 = requests.post(
                    f"{backend_url}/find-chunks",
                    json={
                        "chunks": st.session_state.chunks,
                        "question": user_question,
                        "top_k": 3
                    },
                    timeout=60
                )
                
                if response3.status_code == 200:
                    result3 = response3.json()
                    top_chunks = result3['top_chunks']
                    context = "\n\n".join(top_chunks)
                    
                    # Step 4: Get AI answer
                    response4 = requests.post(
                        f"{backend_url}/ask-gpt",
                        json={
                            'question': user_question,
                            'context': context,
                            'api_key': st.session_state.api_key
                        },
                        timeout=90
                    )
                    
                    if response4.status_code == 200:
                        result4 = response4.json()
                        answer = result4["answer"]
                        
                        # Display answer
                        st.markdown("### 💡 Answer:")
                        st.success(answer)
                        
                        # Show context
                        with st.expander("📜 View Source Context (Click to expand)"):
                            st.markdown("**Retrieved passages from your PDF:**")
                            for i, chunk in enumerate(top_chunks, 1):
                                st.markdown(f"**Passage {i}:**")
                                st.text(chunk)
                                st.markdown("---")
                        
                        # Similarity scores if available
                        if 'similarity_scores' in result3:
                            with st.expander("📊 Relevance Scores"):
                                scores = result3['similarity_scores']
                                for i, score in enumerate(scores, 1):
                                    st.progress(score, text=f"Passage {i}: {score:.2%} relevant")
                    else:
                        st.error(f"❌ Error getting AI response: {response4.status_code}")
                        st.write(response4.json())
                else:
                    st.error(f"❌ Error finding relevant chunks: {response3.status_code}")
            
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The AI might be processing. Try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Made with ❤️ using Streamlit, FastAPI & Google Gemini</p>
        <p>🔒 Your data is processed securely and not stored</p>
    </div>
""", unsafe_allow_html=True)
