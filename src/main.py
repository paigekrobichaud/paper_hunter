from ingest import find_pdfs, load_doc, chunk_doc
from embeddings import get_model, embed_chunks
from search import search_chunks
from storage import save_chunks, load_chunks
from zotero_client import zot_client, get_zot_pdfs, download_zot_pdfs
from pathlib import Path
import argparse

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        prog='paper_hunter',
        description='Semantic search engine for research papers'
    )
    parser.add_argument(
        '--rebuild',
        action = 'store_true', 
        help='Regenerate paper embeddings'
        )
    parser.add_argument(
        '--top-k',
        action = 'store', 
        type = int,
        default = 3,
        help='Number of relevant paper chunks to return'
        )
    parser.add_argument(
        '--zotero',
        action = 'store_true',
        help='Download papers from Zotero'
    )
    args = parser.parse_args()

    # Load model
    print('Loading embedding model...')
    model = get_model()

    path = Path('../data/chunk_index.pkl')

    # Check if chunks need to be created or rebuilt, otherwise load them 
    if path.exists() and not args.rebuild:
        print('Loading cached data...\n')
        chunks = load_chunks(path)
    else:
        if args.zotero:
            print('Reading papers from Zotero...\n')
            zot = zot_client()
            pdfs = get_zot_pdfs(zot)
            papers = download_zot_pdfs(zot, pdfs)
        else:
            print('Reading papers from local folder...\n')
            papers = find_pdfs()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = []

        print('Embedding data...\n')

        for paper in papers:
            doc = load_doc(paper)
            chunks.extend(chunk_doc(doc))
    
        chunks = embed_chunks(chunks=chunks, model=model)
        save_chunks(chunks,path)

    # Get query and return relevant chunks
    while True:
        query = input('Search query (\'quit\' to exit): ')
        
        if query.lower().strip() == 'quit':
            break
        if not query.strip():
            continue    
        
        relevant_chunks = search_chunks(query=query,chunks=chunks,model=model,top_k=args.top_k)

        for ch, score in relevant_chunks:
            print(f'Article title: {ch.doc.title}\n')
            print(f'Similarity: {score:.3f}\n')
            print(f'Text: {ch.text[:500]}\n')
            print('=================================================================================\n')
    
if __name__=='__main__':
    main()

    