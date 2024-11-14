from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import os
import json
from openai import OpenAI
from typing import List, Dict, Tuple
import numpy as np

app = FastAPI(title="Semantic Search API")
client = OpenAI()

class SearchQuery(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    urls: List[str]
    scores: List[float]

# Global variables for index and metadata
index = None
metadata = None

def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def load_index_and_metadata(data_dir: str) -> Tuple[faiss.Index, Dict]:
    """Load the FAISS index and metadata mapping"""
    index_path = os.path.join(data_dir, "markdown_embeddings.faiss")
    metadata_path = os.path.join(data_dir, "metadata_mapping.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    index = faiss.read_index(str(index_path))
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return index, metadata

@app.on_event("startup")
async def startup_event():
    """Load the FAISS index and metadata on startup"""
    global index, metadata
    try:
        index, metadata = load_index_and_metadata("data")
        print(f"Loaded index with {index.ntotal} vectors")
    except Exception as e:
        print(f"Error loading index and metadata: {str(e)}")
        raise

@app.post("/search", response_model=SearchResult)
async def search(query: SearchQuery) -> SearchResult:
    """Search endpoint"""
    if index is None or metadata is None:
        raise HTTPException(status_code=500, detail="Search index not initialized")
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query.query)
        
        # Search the index
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_vector, query.k)
        
        # Process results
        urls = []
        scores = []
        seen_sources = set()
        
        for distance, doc_idx in zip(distances[0], indices[0]):
            if doc_idx == -1:
                continue
                
            doc_metadata = metadata[str(doc_idx)]
            source = doc_metadata['source']
            
            if source not in seen_sources:
                url = source.replace('/Users/stefan/work/dyalog-docs/documentation/', 
                                  'https://dyalog.github.io/documentation/20.0/')
                url = url.replace('/docs/', '/')
                url = url.replace('.md', '')
                urls.append(url)
                scores.append(float(distance))
                seen_sources.add(source)
        
        return SearchResult(urls=urls, scores=scores)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable is not set")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)