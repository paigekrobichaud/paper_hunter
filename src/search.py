
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import numpy as np

from document import Document
from chunk import Chunk

def search_chunks(query: str, chunks: list[Chunk], model: SentenceTransformer, top_k: int=5)->list[tuple[Chunk,float]]:
    ''' Search for relevant chunks based on query using cosine similarity

    Args:
        query (str):
            User given query for content
        chunks (list[Chunk]):
            Segments from documents
        top_k (int):
            Number of relevant chunks to pull
    
    Returns:
        list[Chunk]:
            top_k most similar chunks to query
    '''

    query_embed = np.array(model.encode([query])) # needs to be 2D array for cosine_sim
    embeddings = np.vstack([ch.embedding for ch in chunks])

    similarity = cosine_similarity(query_embed,embeddings).squeeze()

    top_idx = similarity.argsort()[::-1][:top_k]

    return [(chunks[idx],similarity[idx]) for idx in top_idx] 