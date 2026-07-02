from pathlib import Path
from dataclasses import dataclass

@dataclass

class Document:
    '''Represents a reseach paper
    Attributes:
        path (Path):
            Where the paper is located
        text (str):
            Text inside the document
        title (str):
            Title of the paper
    '''
    path: Path
    text: str
    title: str = '' 

    def __post_init__(self):
    # initialize fields 
        if not self.title:
            self.title = self.path.stem
