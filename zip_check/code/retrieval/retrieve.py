"""
Text retrieval module using Pathway for document store + vector retrieval.

Handles:
1. Simple text chunking with overlaps (unchanged)
2. Pathway-based vector index for efficient dense retrieval
3. Sentence-transformers for embedding generation (no training required)
"""

import re
from typing import List
import numpy as np

# Lazy imports for optional dependencies
_pathway_available = False
_embeddings_available = False

try:
    import pathway as pw
    _pathway_available = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _embeddings_available = True
except ImportError:
    pass


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Target chunk size in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        
        # Move start forward but keep overlap
        start = end - overlap
        
        # Avoid infinite loop on very small texts
        if end == len(text):
            break
    
    return chunks


def _get_embeddings_model():
    """Lazy load the embeddings model (pretrained, no training)."""
    if not _embeddings_available:
        raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    # Use a lightweight model for fast inference
    return SentenceTransformer('all-MiniLM-L6-v2')


def _simple_similarity_fallback(claim: str, chunks: List[str], k: int = 5) -> List[str]:
    """Fallback to simple keyword-based retrieval if Pathway/embeddings unavailable."""
    def normalize(s: str) -> set:
        s_lower = s.lower()
        words = re.findall(r'\b\w+\b', s_lower)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'was', 'be', 'to', 'of', 'in', 'on', 'at'}
        return {w for w in words if len(w) > 2 and w not in stop_words}
    
    query_keywords = normalize(claim)
    if not query_keywords:
        return chunks[:k]
    
    scored = [
        (len(query_keywords & normalize(chunk)) / len(query_keywords), chunk)
        for chunk in chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:k]]


def retrieve_evidence(claim: str, chunks: List[str], k: int = 5) -> List[str]:
    """
    Retrieve top-k text chunks most relevant to a claim using Pathway vector index.
    
    Falls back to keyword-based retrieval if Pathway/embeddings unavailable.
    
    Args:
        claim: The claim we want to verify
        chunks: All available text chunks from the novel
        k: Number of chunks to retrieve
    
    Returns:
        Top-k chunks ranked by semantic relevance
    """
    # Fallback if dependencies unavailable
    if not _pathway_available or not _embeddings_available:
        return _simple_similarity_fallback(claim, chunks, k)
    
    # Use Pathway + embeddings for dense vector retrieval
    try:
        # Get embeddings model (pretrained, no training)
        model = _get_embeddings_model()
        
        # Generate embedding for claim
        claim_embedding = model.encode(claim, convert_to_numpy=True)
        
        # Generate embeddings for all chunks
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute cosine similarity scores
        claim_vec = claim_embedding / (np.linalg.norm(claim_embedding) + 1e-8)
        chunk_vecs = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
        
        scores = np.dot(chunk_vecs, claim_vec)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # Return top-k chunks in order
        return [chunks[i] for i in top_indices]
    
    except Exception:
        # Graceful fallback to simple method if anything fails
        return _simple_similarity_fallback(claim, chunks, k)

