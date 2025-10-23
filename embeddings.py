import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import os


class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str = None):
        print(f"Loading embedding model: {model_name}")
        print("(First time will download ~80MB, then cached)")
        
        # Set cache folder if provided
        if cache_folder:
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_folder
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            print(f"✓ Model loaded successfully")
            print(f"  - Embedding dimension: {self.dimension}")
            print(f"  - Max sequence length: {self.model.max_seq_length}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 32, 
                       show_progress: bool = True) -> np.ndarray:
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        # Clean texts (empty strings cause issues)
        texts = [t.strip() if t else " " for t in texts]
        
        try:
            # Generate embeddings (uses GPU if available)
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Use the same encoding function for consistency
        embedding = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]  # Get first (and only) embedding
        
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        # Since embeddings are normalized, dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.dimension,
            'max_sequence_length': self.model.max_seq_length,
            'device': str(self.model.device)
        }


# Utility function for similarity matrix
def compute_similarity_matrix(embeddings1: np.ndarray, 
                             embeddings2: np.ndarray) -> np.ndarray:
    # Normalized embeddings → dot product = cosine similarity
    similarities = np.dot(embeddings1, embeddings2.T)
    return similarities


# Example usage and testing
if __name__ == "__main__":
    # Test the embedding generator
    print("Testing Embedding Generator\n")
    
    generator = EmbeddingGenerator()
    
    # Test documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks",
        "Python is a programming language",
        "AI is transforming the world"
    ]
    
    # Generate embeddings
    print("\nGenerating embeddings for documents...")
    doc_embeddings = generator.embed_documents(documents, show_progress=True)
    print(f"Shape: {doc_embeddings.shape}")
    
    # Test query
    query = "What is AI?"
    print(f"\nQuery: {query}")
    query_embedding = generator.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Compute similarities
    print("\nSimilarity scores:")
    for i, doc in enumerate(documents):
        similarity = generator.compute_similarity(query_embedding, doc_embeddings[i])
        print(f"  {doc[:50]}... → {similarity:.4f}")
    
    # Model info
    print("\nModel Info:")
    info = generator.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")