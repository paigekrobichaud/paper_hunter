from pathlib import Path

import pymupdf

from document import Document
from chunk import Chunk


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
        papers_dir = project_root / 'papers/paper_hunter_papers_backup/papers/'

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