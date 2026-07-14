from document import Document

from dataclasses import dataclass
import numpy as np

@dataclass
class Chunk:
    '''
    Represents a chunk of text
    Attributes:
        text (str):
            Text from the paper
        embedding (np.array):
            SentenceTransformer embedding
        doc (Document):
            Document class instance of document where text is from
        idx (int):
            Identifies chunk within doc
    '''
    text: str
    embedding: np.array
    doc: Document
    idx: int