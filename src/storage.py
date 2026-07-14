## Be able to store chunks so that embedding / chunking only needs to happen once

from chunk import Chunk
from pathlib import Path

import pickle

def save_chunks(chunks: list[Chunk], path: Path)->None: 
    with open(path,'wb') as file: 
        pickle.dump(chunks,file)

def load_chunks(path: Path)->list[Chunk]:
    with open(path,'rb') as file:
        chunks = pickle.load(file)

    return chunks