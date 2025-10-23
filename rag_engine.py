import os
from typing import List, Dict
from pathlib import Path
import ollama

from .document_loader import DocumentLoader
from .chunker import SemanticChunker
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore


class RAGEngine:
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "llama3.2",
                 persist_directory: str = "./chroma_db"):
        print("🚀 Initializing MindVault RAG Engine\n")
        
        # Initialize components
        print("1/4 Loading document processor...")
        self.document_loader = DocumentLoader()
        
        print("2/4 Initializing text chunker...")
        self.chunker = SemanticChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        
        print("3/4 Loading embedding model...")
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        print("4/4 Connecting to vector store...")
        self.vector_store = VectorStore(persist_directory=persist_directory)
        
        # LLM configuration (Ollama only)
        self.llm_model = llm_model
        
        print("\n✅ RAG Engine ready!\n")
    
    def ingest_document(self, file_path: str) -> Dict:
        print(f"📄 Ingesting: {Path(file_path).name}")
        
        try:
            # Step 1: Load document
            print("  → Loading document...")
            doc_data = self.document_loader.load_document(file_path)
            
            # Step 2: Chunk text
            print("  → Chunking text...")
            chunks = self.chunker.chunk_document(
                text=doc_data['text'],
                filename=doc_data['filename'],
                file_type=doc_data['file_type']
            )
            
            # Step 3: Generate embeddings
            print(f"  → Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_generator.embed_documents(
                chunk_texts,
                show_progress=False
            )
            
            # Step 4: Store in vector database
            print("  → Storing in vector database...")
            self.vector_store.add_documents(chunks, embeddings)
            
            stats = {
                'filename': doc_data['filename'],
                'file_type': doc_data['file_type'],
                'text_length': len(doc_data['text']),
                'chunk_count': len(chunks),
                'success': True
            }
            
            print(f"✅ Successfully ingested {doc_data['filename']}")
            print(f"   Created {len(chunks)} chunks\n")
            
            return stats
            
        except Exception as e:
            print(f"❌ Error ingesting document: {e}\n")
            return {
                'filename': Path(file_path).name,
                'success': False,
                'error': str(e)
            }
    
    def query(self, question: str, top_k: int = 3, 
              use_hybrid_search: bool = True) -> Dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")     
        
        # Check if we have documents
        if self.vector_store.collection.count() == 0:
            return {
                'answer': "I don't have any documents in my knowledge base yet. Please upload some documents first.",
                'sources': [],
                'metadata': {'retrieved_chunks': 0}
            }
        
        print(f"🔍 Processing query: {question}\n")
        
        try:
            # Step 1: Generate query embedding
            print("  → Generating query embedding...")
            query_embedding = self.embedding_generator.embed_query(question)
            
            # Step 2: Search vector database
            print(f"  → Searching for top {top_k} relevant chunks...")
            if use_hybrid_search:
                retrieved_chunks = self.vector_store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=question,
                    top_k=top_k
                )
            else:
                retrieved_chunks = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            
            if not retrieved_chunks:
                return {
                    'answer': "I couldn't find any relevant information in the documents.",
                    'sources': [],
                    'metadata': {'retrieved_chunks': 0}
                }
            
            # Step 3: Generate answer using Ollama
            print(f"  → Generating answer using Ollama ({self.llm_model})...")
            answer = self._generate_answer(question, retrieved_chunks)
            
            # Format sources
            sources = self._format_sources(retrieved_chunks)
            
            # Calculate metadata
            avg_similarity = sum(c.get('similarity_score', 0) for c in retrieved_chunks) / len(retrieved_chunks)
            
            result = {
                'answer': answer,
                'sources': sources,
                'metadata': {
                    'retrieved_chunks': len(retrieved_chunks),
                    'avg_similarity': round(avg_similarity, 3),
                    'search_type': 'hybrid' if use_hybrid_search else 'semantic',
                    'llm_model': self.llm_model
                }
            }
            
            print("✅ Answer generated successfully\n")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}\n")
            return {
                'answer': f"Error processing your question: {str(e)}",
                'sources': [],
                'metadata': {'error': str(e)}
            }
    
    def _generate_answer(self, question: str, chunks: List[Dict]) -> str:
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Source {i}] {chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Construct prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- Be specific and cite which source you're using (e.g., "According to Source 1...")
- If the context doesn't contain enough information, say so clearly
- Keep your answer concise but complete
- Do not make up information

Answer:"""
        
        # Generate with Ollama
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'num_predict': 500
                }
            )
            return response['response'].strip()
            
        except Exception as e:
            return f"Error with Ollama: {str(e)}. Make sure Ollama is running with 'ollama serve'"
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format retrieved chunks as sources"""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append({
                'source_number': i,
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'filename': chunk['metadata']['filename'],
                'chunk_index': chunk['metadata']['chunk_index'],
                'similarity_score': round(chunk.get('similarity_score', 0), 3)
            })
        return sources
    
    def delete_document(self, filename: str) -> Dict:
        """Delete a document from the knowledge base"""
        try:
            deleted_count = self.vector_store.delete_document(filename)
            return {
                'success': True,
                'deleted_chunks': deleted_count,
                'filename': filename
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename
            }
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents"""
        self.vector_store.clear_all()
        print("🗑️  Knowledge base cleared")
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        vector_stats = self.vector_store.get_stats()
        embedding_info = self.embedding_generator.get_model_info()
        
        return {
            'vector_store': vector_stats,
            'embedding_model': embedding_info,
            'llm_model': self.llm_model,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.overlap
        }
    
    def list_documents(self) -> List[Dict]:
        """List all documents in knowledge base"""
        return self.vector_store.get_documents_list()


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = RAGEngine(llm_model="llama3.2")
    
    # Test ingestion (create a test file first)
    test_file = "test_doc.txt"
    with open(test_file, 'w') as f:
        f.write("Machine learning is a subset of artificial intelligence. "
                "It focuses on teaching computers to learn from data.")
    
    # Ingest
    result = engine.ingest_document(test_file)
    print(f"Ingestion result: {result}")
    
    # Query
    answer = engine.query("What is machine learning?")
    print(f"\nQuestion: What is machine learning?")
    print(f"Answer: {answer['answer']}")
    print(f"Sources: {len(answer['sources'])}")
    
    # Cleanup
    os.remove(test_file)
    engine.clear_knowledge_base()
