
from document import Document
from chunk import Chunk

from sentence_transformers import SentenceTransformer


def get_model(model_type: str='')->SentenceTransformer:
    ''' Get sentence transformer model to do embeddings
    
    Args:
        model_type (str):
            Specifies model within the SentenceTransformer class
    
    Returns:
        SentenceTransformer:
            Loads instance of a SentenceTransformer model
    '''
    
    if not model_type:
        # default to MiniLM L6 - fast, not too big
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    else:
        return SentenceTransformer(model_type)

def embed_chunks(chunks: list[Chunk], model: SentenceTransformer)->list[Chunk]:
    '''Creates an embedding for the text in chunk
    
    Args:
        chunks (list[Chunk]):
            List of instances of the Chunk class to embed
    
    Returns:
        list[Chunk]
            Embedded chunks
    '''

    chunks_text = [ch.text for ch in chunks]
    embeds = model.encode(chunks_text)

    for ch, embedding in zip(chunks,embeds):
        ch.embedding = embedding

    return chunks