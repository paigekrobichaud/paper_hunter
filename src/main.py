from ingest import find_pdfs, load_doc, chunk_doc
from embeddings import get_model, embed_chunks
from search import search_chunks
from storage import save_chunks, load_chunks
from pathlib import Path

def main():

    print('Loading embedding model...')
    model = get_model()

    path = Path('../data/chunk_index.pkl')

    # Check if chunks need to be created, otherwise load them 
    if path.exists():
        print('Loading cached data...\n')
        chunks = load_chunks(path)
    else:
        print('Reading papers and embedding data...\n')
        path.parent.mkdir(parents=True, exist_ok=True)
        papers = find_pdfs()
        chunks = []

        for paper in papers:
            doc = load_doc(paper)
            chunks.extend(chunk_doc(doc))
    
        chunks = embed_chunks(chunks=chunks, model=model)
        save_chunks(chunks,path)

    while True:
        query = input('Key word(s) search (\'quit\' to exit): ')
        
        if query.lower().strip() == 'quit':
            break
        if not query.strip():
            continue    
        
        relevant_chunks = search_chunks(query=query,chunks=chunks,model=model,top_k=3)

        for ch, score in relevant_chunks:
            print(f'Article title: {ch.doc.title}\n')
            print(f'Similarity: {score:.3f}\n')
            print(f'Text: {ch.text[:500]}\n')
            print('=================================================================================\n')
    
if __name__=='__main__':
    main()

    