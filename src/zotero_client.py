import os
from pyzotero import zotero
from dotenv import load_dotenv
from pprint import pprint
from pathlib import Path

def zot_client()->zotero:
    '''
    Create a Zotero client to interface with
    Assumes Zotero library info in .env
    
    Returns:
        zotero instance
    '''
    load_dotenv()

    library_id = os.getenv('ZOTERO_LIBRARY_ID')
    library_type = os.getenv('ZOTERO_LIBRARY_TYPE')
    api_key = os.getenv('ZOTERO_API_KEY')

    return zotero.Zotero(library_id, library_type, api_key)

def get_zot_pdfs(zot: zotero, num=10)->list[dict]:
    '''
    Find top items in Zotero library

    Args:
        zot: Zotero
        num: int
            Number of top items to return
    
    Returns:
        list[dict]
            Corresponds to items in Zotero library
    '''
    return zot.top(limit=num)

def download_zot_pdfs(zot: zotero, items: list[dict], download_dir=None)->list[Path]:
    '''
    Download files from Zotero

    Args:
        zot: Zotero
        items: list[dict]
            Items from Zotero library to download
        download_dir: Path
            Specifies where to download items

    Returns: 
        list[Path]:
            pdf paths
    '''
    if not download_dir:
        download_dir = Path('../papers/zotero_papers')
    download_dir.mkdir(parents=True, exist_ok = True)

    paths = []

    for item in items:
        data = item.get('data',{})
        children = zot.children(item['key'])

        for child in children:
            child_data = child.get('data',{})

            if (
                child_data.get('itemType') == 'attachment'
                and child_data.get('contentType') == 'application/pdf'
            ):
                key = child['key']
                filename = child_data.get('filename')

                # download the pdfs
                zot.dump(
                    key,
                    filename=filename,
                    path=download_dir
                )

                # save the paths
                paths.append(download_dir / filename)

    return paths
    

    
    