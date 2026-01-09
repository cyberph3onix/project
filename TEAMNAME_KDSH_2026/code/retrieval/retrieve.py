"""
Text retrieval module for RAG-style evidence extraction.

Handles:
1. Simple text chunking with overlaps
2. Keyword-based similarity retrieval (can be swapped for dense embeddings)
"""

import re
from typing import List, Tuple


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


def compute_keyword_overlap(text1: str, text2: str) -> float:
    """
    Simple similarity metric: fraction of keywords from text1 found in text2.
    
    This is a naive baseline. In production, replace with:
    - Cosine similarity of TF-IDF vectors
    - Embedding-based similarity (dense retrieval)
    - BM25 ranking
    
    Args:
        text1: Query/claim text
        text2: Candidate chunk text
    
    Returns:
        Overlap score in [0, 1]
    """
    # Normalize: lowercase, remove punctuation, split to keywords
    def normalize(s: str) -> set:
        s_lower = s.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', s_lower)
        # Filter out stop words (minimal list for simplicity)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'was', 'be', 'to', 'of', 'in', 'on', 'at'}
        return {w for w in words if len(w) > 2 and w not in stop_words}
    
    query_keywords = normalize(text1)
    candidate_keywords = normalize(text2)
    
    if not query_keywords:
        return 0.0
    
    # Intersection / size of query set = recall-like metric
    overlap = len(query_keywords & candidate_keywords)
    return overlap / len(query_keywords)


def retrieve_evidence(claim: str, chunks: List[str], k: int = 5) -> List[str]:
    """
    Retrieve top-k text chunks most relevant to a claim.
    
    Args:
        claim: The claim we want to verify
        chunks: All available text chunks from the novel
        k: Number of chunks to retrieve
    
    Returns:
        Top-k chunks ranked by relevance
    """
    # Rank chunks by similarity to claim
    scored_chunks = [
        (compute_keyword_overlap(claim, chunk), chunk)
        for chunk in chunks
    ]
    
    # Sort descending by score and take top-k
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top_k = scored_chunks[:k]
    
    # Return just the chunk text (not scores)
    return [chunk for _, chunk in top_k]
