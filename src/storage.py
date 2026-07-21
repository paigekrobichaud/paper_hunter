## Be able to store chunks so that embedding / chunking only needs to happen once

from chunk import Chunk
from pathlib import Path

import pickle

def save_chunks(chunks: list[Chunk], path: Path)->None: 
    '''
    Save a list of chunks to path, for loading later
    
    Args:
        chunks (list[Chunk]):
            List of Chunks to save
        path (Path):
            Where to save chunks
    '''
    with open(path,'wb') as file: 
        pickle.dump(chunks,file)

def load_chunks(path: Path)->list[Chunk]:
    '''
    Save a list of chunks to path, for loading later
    
    Args:
        path (Path):
            Where to retrieve chunks
    Returns:
        list[Chunk]
            Saved chunks
    '''
    with open(path,'rb') as file:
        chunks = pickle.load(file)

    return chunks