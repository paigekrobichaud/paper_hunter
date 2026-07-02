from document import Document

from dataclasses import dataclass
import numpy as np

@dataclass
class Chunk:
    text: str
    embedding: np.array
    doc: Document
    idx: int