import re
from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ChunkMetadata:
    chunk_id: str
    filename: str
    file_type: str
    chunk_index: int
    total_chunks: int
    char_count: int
    created_at: str


class SemanticChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str, filename: str, file_type: str) -> List[Dict]:
        # Clean and validate text
        text = self._clean_text(text)
        
        if len(text) < self.chunk_size:
            # Document is smaller than chunk size - return as single chunk
            return [self._create_chunk(
                text=text,
                filename=filename,
                file_type=file_type,
                chunk_index=0,
                total_chunks=1
            )]
        
        # Split into sentences (core of semantic chunking)
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            raise ValueError("No sentences found in text")
        
        # Build chunks respecting sentence boundaries
        chunks = self._build_chunks(sentences, filename, file_type)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        # Pattern matches sentence endings followed by space or newline
        # Handles: ., !, ?, and combinations with quotes
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(pattern, text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _build_chunks(self, sentences: List[str], filename: str, file_type: str) -> List[Dict]:
        chunks = []
        current_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds chunk_size
            if current_length + sentence_length > self.chunk_size and current_sentences:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_sentences)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    filename=filename,
                    file_type=file_type,
                    chunk_index=len(chunks),
                    total_chunks=-1  # Will update later
                ))
                
                # Calculate overlap: keep last N sentences
                overlap_text_length = 0
                overlap_sentences = []
                
                # Go backwards to find sentences that fit in overlap
                for sent in reversed(current_sentences):
                    if overlap_text_length + len(sent) <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_text_length += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap + current sentence
                current_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_sentences)
            else:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(self._create_chunk(
                text=chunk_text,
                filename=filename,
                file_type=file_type,
                chunk_index=len(chunks),
                total_chunks=-1
            ))
        
        # Update total_chunks in all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total
        
        return chunks
    
    def _create_chunk(self, text: str, filename: str, file_type: str, 
                     chunk_index: int, total_chunks: int) -> Dict:
        """Create chunk dictionary with text and metadata"""
        chunk_id = f"{filename}_chunk_{chunk_index}"
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            filename=filename,
            file_type=file_type,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            char_count=len(text),
            created_at=datetime.now().isoformat()
        )
        
        return {
            'text': text,
            'metadata': asdict(metadata)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    Artificial intelligence is transforming the world. Machine learning is a subset of AI.
    Deep learning uses neural networks. These networks have multiple layers.
    Each layer learns different features. This creates powerful models.
    Natural language processing is another AI field. It helps computers understand human language.
    """
    
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(sample_text, "test.txt", "txt")
    
    print(f"Created {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")
        print()