import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Import our RAG engine
from src.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MindVault - Intelligent Knowledge Base",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-message {
        padding: 1rem;
        background-color: #d1fae5;
        color: #065f46;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        padding: 1rem;
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_engine():
    return RAGEngine(
        chunk_size=int(os.getenv('CHUNK_SIZE', 500)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50)),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        llm_provider=os.getenv('LLM_PROVIDER', 'ollama'),
        llm_model=os.getenv('OLLAMA_MODEL', 'llama3.2'),
        groq_api_key=os.getenv('GROQ_API_KEY'),
        persist_directory=os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
    )


def main():
    
    # Header
    st.markdown('<div class="main-header">🧠 MindVault</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your Intelligent Knowledge Base powered by RAG</div>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    try:
        rag_engine = init_rag_engine()
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        st.info("Make sure Ollama is running if you're using local LLM: `ollama serve`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # System stats
        st.subheader("📊 System Status")
        stats = rag_engine.get_stats()
        st.metric("Total Chunks", stats['vector_store']['total_chunks'])
        st.metric("Documents", stats['vector_store']['unique_documents'])
        st.metric("Embedding Dim", stats['embedding_model']['embedding_dimension'])
        
        st.divider()
        
        # Query settings
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
        use_hybrid = st.checkbox("Use Hybrid Search", value=True, 
                                help="Combines semantic and keyword search")
        
        st.divider()
        
        # LLM info
        st.subheader("🤖 LLM Configuration")
        st.text(f"Provider: {stats['llm_provider']}")
        st.text(f"Model: {stats['llm_model']}")
        
        st.divider()
        
        # Danger zone
        st.subheader("⚠️ Danger Zone")
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                rag_engine.clear_knowledge_base()
                st.success("All documents cleared!")
                st.session_state['confirm_clear'] = False
                st.rerun()
            else:
                st.session_state['confirm_clear'] = True
                st.warning("Click again to confirm")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📤 Upload Documents", "📚 Knowledge Base"])
    
    # Tab 1: Query Interface
    with tab1:
        st.header("Ask Your Documents")
        
        if stats['vector_store']['total_chunks'] == 0:
            st.warning("⚠️ No documents in knowledge base. Upload some documents first!")
        else:
            # Query input
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the key points about machine learning?",
                key="query_input"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                search_button = st.button("🔍 Search", type="primary", use_container_width=True)
            
            # Process query
            if search_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = rag_engine.query(
                            question=question,
                            top_k=top_k,
                            use_hybrid_search=use_hybrid
                        )
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.markdown(f"**{result['answer']}**")
                        
                        # Display metadata
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Chunks Retrieved", result['metadata']['retrieved_chunks'])
                        with col2:
                            st.metric("Avg Similarity", f"{result['metadata']['avg_similarity']:.3f}")
                        with col3:
                            st.metric("Search Type", result['metadata']['search_type'].title())
                        
                        # Display sources
                        if result['sources']:
                            st.divider()
                            st.subheader("📚 Sources")
                            for source in result['sources']:
                                with st.expander(f"Source {source['source_number']}: {source['filename']} (Similarity: {source['similarity_score']})"):
                                    st.markdown(f"**Chunk {source['chunk_index']}**")
                                    st.text(source['text'])
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
    
    # Tab 2: Upload Interface
    with tab2:
        st.header("Upload Documents")
        
        st.info("📝 Supported formats: .txt, .pdf, .docx, .md")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary"):
                # Create temp directory for uploads
                temp_dir = Path("./temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file temporarily
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Ingest document
                    result = rag_engine.ingest_document(str(temp_path))
                    results.append(result)
                    
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Cleanup temp file
                    temp_path.unlink()
                
                # Cleanup temp directory
                temp_dir.rmdir()
                
                # Show results
                status_text.empty()
                progress_bar.empty()
                
                success_count = sum(1 for r in results if r['success'])
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count}/{len(results)} documents!")
                    
                    # Show details
                    with st.expander("View Details"):
                        for result in results:
                            if result['success']:
                                st.markdown(f"""
                                <div class="success-message">
                                    <strong>{result['filename']}</strong><br>
                                    Type: {result['file_type']} | 
                                    Chunks: {result['chunk_count']} | 
                                    Characters: {result['text_length']:,}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="error-message">
                                    <strong>{result['filename']}</strong><br>
                                    Error: {result['error']}
                                </div>
                                """, unsafe_allow_html=True)
                
                # Force refresh
                st.rerun()
    
    # Tab 3: Knowledge Base Management
    with tab3:
        st.header("Knowledge Base")
        
        documents = rag_engine.list_documents()
        
        if not documents:
            st.info("📭 No documents in knowledge base yet. Upload some documents to get started!")
        else:
            st.subheader(f"📚 {len(documents)} Document(s) in Knowledge Base")
            
            # Display as table
            for doc in documents:
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 2, 1])
                
                with col1:
                    st.text(f"📄 {doc['filename']}")
                with col2:
                    st.text(f"{doc['file_type']}")
                with col3:
                    st.text(f"{doc['chunk_count']} chunks")
                with col4:
                    st.text(f"{doc['created_at'][:10]}")
                with col5:
                    if st.button("🗑️", key=f"delete_{doc['filename']}", help="Delete document"):
                        result = rag_engine.delete_document(doc['filename'])
                        if result['success']:
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
                        else:
                            st.error(f"Error: {result['error']}")
                
                st.divider()
            
            # Bulk actions
            st.subheader("Bulk Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Chunks", sum(d['chunk_count'] for d in documents))
            with col2:
                total_size = stats['vector_store']['total_chunks']
                st.metric("Vector Store Size", f"{total_size:,}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p><strong>MindVault</strong> - Built with ❤️ using RAG Technology</p>
        <p style="font-size: 0.9rem;">
            Powered by: sentence-transformers · ChromaDB · {llm_provider}
        </p>
    </div>
    """.format(llm_provider=stats['llm_provider'].title()), unsafe_allow_html=True)


if __name__ == "__main__":
    main()