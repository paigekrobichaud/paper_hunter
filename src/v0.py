## Ingest summary

from pathlib import Path

import pymupdf

from document import Document
from chunk import Chunk

from sentence_transformers import SentenceTransformer

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity



# ----------------------------
# Document ingestion
# ----------------------------

def find_pdfs(papers_dir=None)->list[Path]:
    ''' Find all pdf files in the papers directory

    Args:
        papers_dir (Path, optional):
            Directory containing pdf files. If None, defaults to 
            the project's 'papers' directory

    Returns:
        list[Path]:
            A list of paths to all pdf files found in 'papers_dir'

    '''
    if not papers_dir:
        project_root = Path(__file__).parent.parent
        papers_dir = project_root / 'papers'

    return list(papers_dir.glob('*.pdf'))

def load_doc(paper_path: Path)->Document:
    '''Return document instance for document in paper_path
    
    Args:
        paper_path (Path):
            Path to pdf
        
    Returns:
        document:
            class containing text, title, and path
    '''
    with pymupdf.open(paper_path) as doc:
        text = ''
        for page in doc:
            text += page.get_text()
    
    return Document(path=paper_path,text=text)
    
def chunk_doc(doc: Document, chunk_size: int=500)-> list[Chunk]:
    ''' Split up document into several chunks of chunk_size
    
    Args:
        doc (Document):
            Instance of document class to chunk text
        chunk_size (int):
            Number of words per chunk
    
    Returns:
        list[Chunk]:
            List of Chunks of the document text
    '''

    split_text = doc.text.split()
    chunks = []

    for idx, i in enumerate(range(0, len(split_text), chunk_size)):
        word_chunk = split_text[i:i+chunk_size]
        chunk_str = ' '.join(word_chunk)
        new_chunk = Chunk(text=chunk_str,embedding=None,doc=doc,idx=idx)
        
        chunks.append(new_chunk)

    return chunks

# ----------------------------
# Embedding
# ----------------------------

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

# ----------------------------
# Retrieval
# ----------------------------

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
    embeddings = np.array([ch.embedding for ch in chunks])

    similarity = cosine_similarity(query_embed,embeddings).squeeze()

    top_idx = similarity.argsort()[::-1][:top_k]

    return [(chunks[idx],similarity[idx]) for idx in top_idx] 

if __name__=='__main__':
    
    
    papers = find_pdfs()

    chunks = []

    query = 'parity-doublet states'

    for paper in papers:
        doc = load_doc(paper)
        chunks.extend(chunk_doc(doc))
    
    model = get_model()
    chunks = embed_chunks(chunks=chunks, model=model)


    relevant_chunks = search_chunks(query=query,chunks=chunks,model=model,top_k=3)

    for ch, score in relevant_chunks:
        print(f'Article title: {ch.doc.title}\n')
        print(f'Similarity: {score}\n')
        print(f'Text: {ch.text[:500]}\n')


    

    
    

    