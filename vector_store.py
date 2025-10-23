import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path


class VectorStore: 
    def __init__(self, persist_directory: str = "./chroma_db", 
                 collection_name: str = "mindvault_docs"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing vector store at: {persist_directory}")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True
            )
        )
        
        # Get or create collection
        # distance: cosine similarity (standard for embeddings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # HNSW algorithm for fast search
        )
        
        doc_count = self.collection.count()
        print(f"✓ Vector store ready. Documents: {doc_count}")
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if len(chunks) == 0:
            raise ValueError("Cannot add empty document list")
        
        # Prepare data for ChromaDB
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        embeddings_list = embeddings.tolist()  # ChromaDB expects lists
        
        try:
            # Add to collection (batch operation)
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            
            print(f"✓ Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            raise RuntimeError(f"Error adding documents to vector store: {e}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.collection.count()),
                where=filter_metadata,  # Apply filters if provided
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances (lower = more similar)
                # Convert to similarity score (higher = more similar)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Cosine distance to similarity
                
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': float(similarity)
                })
            
            return formatted_results
            
        except Exception as e:
            raise RuntimeError(f"Error searching vector store: {e}")
    
    def hybrid_search(self, query_embedding: np.ndarray, query_text: str,
                     top_k: int = 5, semantic_weight: float = 0.7) -> List[Dict]:

        # Get more candidates for reranking
        candidates = self.search(query_embedding, top_k=top_k * 2)
        
        if not candidates:
            return []
        
        # Extract keywords from query
        query_keywords = set(query_text.lower().split())
        
        # Rerank with hybrid scoring
        for result in candidates:
            semantic_score = result['similarity_score']
            
            # Simple keyword matching (BM25-like)
            doc_words = set(result['text'].lower().split())
            keyword_matches = len(query_keywords & doc_words)
            keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # Combine scores
            hybrid_score = (semantic_weight * semantic_score + 
                          (1 - semantic_weight) * keyword_score)
            
            result['hybrid_score'] = hybrid_score
            result['keyword_score'] = keyword_score
        
        # Sort by hybrid score
        candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return candidates[:top_k]
    
    def delete_document(self, filename: str) -> int:
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if not results['ids']:
                return 0
            
            # Delete chunks
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            print(f"✓ Deleted {deleted_count} chunks from {filename}")
            
            return deleted_count
            
        except Exception as e:
            raise RuntimeError(f"Error deleting document: {e}")
    
    def clear_all(self) -> None:
        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.collection_name)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print("✓ All documents deleted")
            
        except Exception as e:
            raise RuntimeError(f"Error clearing vector store: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        count = self.collection.count()
        
        # Get unique filenames
        if count > 0:
            all_metadata = self.collection.get(include=['metadatas'])
            filenames = set(m['filename'] for m in all_metadata['metadatas'])
            unique_files = len(filenames)
        else:
            unique_files = 0
        
        return {
            'total_chunks': count,
            'unique_documents': unique_files,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }
    
    def get_documents_list(self) -> List[Dict]:
        if self.collection.count() == 0:
            return []
        
        # Get all chunks
        all_data = self.collection.get(include=['metadatas'])
        
        # Group by filename
        docs_dict = {}
        for metadata in all_data['metadatas']:
            filename = metadata['filename']
            if filename not in docs_dict:
                docs_dict[filename] = {
                    'filename': filename,
                    'file_type': metadata['file_type'],
                    'chunk_count': 0,
                    'created_at': metadata['created_at']
                }
            docs_dict[filename]['chunk_count'] += 1
        
        return list(docs_dict.values())


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    print("Testing Vector Store\n")
    
    # Initialize
    store = VectorStore(persist_directory="./test_chroma_db")
    
    # Create sample data
    chunks = [
        {
            'text': 'Machine learning is a subset of AI',
            'metadata': {
                'chunk_id': 'test_0',
                'filename': 'test.txt',
                'file_type': 'txt',
                'chunk_index': 0,
                'total_chunks': 2,
                'char_count': 35,
                'created_at': '2024-01-01'
            }
        },
        {
            'text': 'Deep learning uses neural networks',
            'metadata': {
                'chunk_id': 'test_1',
                'filename': 'test.txt',
                'file_type': 'txt',
                'chunk_index': 1,
                'total_chunks': 2,
                'char_count': 34,
                'created_at': '2024-01-01'
            }
        }
    ]
    
    # Generate dummy embeddings (in real app, use EmbeddingGenerator)
    embeddings = np.random.randn(2, 384)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Add documents
    store.add_documents(chunks, embeddings)
    
    # Search
    query_embedding = np.random.randn(384)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    results = store.search(query_embedding, top_k=2)
    print(f"\nSearch results: {len(results)}")
    for result in results:
        print(f"  - {result['text'][:50]}... (score: {result['similarity_score']:.3f})")
    
    # Stats
    print(f"\nStats: {store.get_stats()}")
    
    # Cleanup
    store.clear_all()
    print("\nCleaned up test database")