"""
Process markdown files into a FAISS vector store
"""

import os
import json
import time
from openai import OpenAI
import tiktoken
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
import faiss
import argparse
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Constants for text splitting and embedding
embedding_encoding = "cl100k_base"  # This is the encoding used by text-embedding-3-small
max_tokens = 8000  # Max tokens per chunk
chunk_overlap = 200  # Number of overlapping tokens between chunks

# Initialize OpenAI client with timeout
client = OpenAI(timeout=30.0)

class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_name="text-embedding-3-small"):
        super().__init__()
        self.model_name = model_name
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((Exception))
    )
    def _get_query_embedding(self, query: str) -> List[float]:
        response = client.embeddings.create(
            input=query,
            model=self.model_name
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((Exception))
    )
    def _get_text_embedding(self, text: str) -> List[float]:
        response = client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

def get_markdown_files(directory):
    """Recursively find all markdown files in directory"""
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.md', '.markdown')):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def get_markdown_files_from_config(config_file):
    """Read directories from config file and collect all markdown files"""
    markdown_files = []
    with open(config_file, 'r') as f:
        directories = [line.strip() for line in f if line.strip()]
    
    for directory in tqdm(directories, desc="Processing directories"):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping")
            continue
        markdown_files.extend(get_markdown_files(directory))
    
    return markdown_files

def create_documents(file_paths):
    """Create Document instances from markdown files"""
    documents = []
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        tokenizer=tiktoken.get_encoding(embedding_encoding).encode
    )
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            print(f"Error reading {file_path}, trying with different encoding")
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception as e:
                print(f"Failed to read {file_path} with alternative encoding: {str(e)}")
                continue
        except Exception as e:
            print(f"Skipping {file_path}: {str(e)}")
            continue
            
        # Split text into chunks if necessary
        chunks = text_splitter.split_text(text)
        
        # Create a Document for each chunk
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                text=chunk,
                metadata={
                    'source': str(file_path),
                    'chunk': i if len(chunks) > 1 else None,
                    'chunk_total': len(chunks)
                }
            ))
    
    return documents

def load_progress(output_dir: str) -> Dict:
    """Load progress from previous run if it exists"""
    progress_file = os.path.join(output_dir, "embedding_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"processed_chunks": [], "embeddings": []}

def save_progress(output_dir: str, processed_chunks: List[Document], embeddings: List[List[float]]):
    """Save current progress to disk"""
    progress_file = os.path.join(output_dir, "embedding_progress.json")
    
    # Convert numpy arrays to lists for JSON serialization
    embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
    
    progress = {
        "processed_chunks": [
            {
                "text": doc.text,
                "metadata": doc.metadata
            } for doc in processed_chunks
        ],
        "embeddings": embeddings_list
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def process_chunks(documents: List[Document], embed_model: OpenAIEmbedding, 
                  output_dir: str, batch_size: int = 10) -> tuple[List[Document], List[List[float]]]:
    """Process documents in batches with progress saving"""
    
    # Load previous progress if it exists
    progress = load_progress(output_dir)
    processed_chunks = []
    embeddings = []
    
    # Convert progress data back to Documents
    if progress["processed_chunks"]:
        processed_chunks = [
            Document(text=chunk["text"], metadata=chunk["metadata"])
            for chunk in progress["processed_chunks"]
        ]
        embeddings = progress["embeddings"]
        print(f"Loaded {len(processed_chunks)} previously processed chunks")
    
    # Process remaining documents
    remaining_docs = documents[len(processed_chunks):]
    if not remaining_docs:
        print("All chunks already processed!")
        return processed_chunks, embeddings
    
    print(f"Processing {len(remaining_docs)} remaining chunks...")
    
    for i in tqdm(range(0, len(remaining_docs), batch_size)):
        batch = remaining_docs[i:i + batch_size]
        batch_embeddings = []
        
        for doc in batch:
            try:
                embedding = embed_model._get_text_embedding(doc.text)
                batch_embeddings.append(embedding)
                processed_chunks.append(doc)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        # Save progress after each batch
        if i % (batch_size * 5) == 0:  # Save every 5 batches
            save_progress(output_dir, processed_chunks, embeddings)
    
    # Final save
    save_progress(output_dir, processed_chunks, embeddings)
    return processed_chunks, embeddings

def create_faiss_index(dimension=1536):
    """Create a simple flat FAISS index"""
    print("Creating flat FAISS index")
    return faiss.IndexFlatL2(dimension)

def main():
    parser = argparse.ArgumentParser(description='Process markdown files into FAISS vector store')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--dir', help='Directory containing markdown files')
    input_group.add_argument('--config', help='File containing list of directories to process')
    parser.add_argument('--max', type=int, help='Maximum number of files to process')
    parser.add_argument('--output', default="data", help='Output directory for index and metadata')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Get and prepare documents
    if args.dir:
        print(f"Finding markdown files in directory: {args.dir}")
        markdown_files = get_markdown_files(args.dir)
    else:
        print(f"Reading directories from config file: {args.config}")
        markdown_files = get_markdown_files_from_config(args.config)
    
    markdown_files = list(dict.fromkeys(markdown_files))
    if args.max:
        markdown_files = markdown_files[:args.max]
    
    print(f"Found {len(markdown_files)} markdown files")
    documents = create_documents(markdown_files)
    print(f"Created {len(documents)} document chunks")
    
    # Process documents and get embeddings
    embed_model = OpenAIEmbedding()
    processed_chunks, embeddings = process_chunks(
        documents, 
        embed_model, 
        args.output,
        args.batch_size
    )
    
    # Create and populate FAISS index
    dimension = 1536
    faiss_index = create_faiss_index(dimension)
    
    # Convert embeddings to numpy array and add to index
    embeddings_array = np.array(embeddings).astype('float32')
    faiss_index.add(embeddings_array)
    
    # Save FAISS index
    index_path = os.path.join(args.output, "markdown_embeddings.faiss")
    print(f"Saving FAISS index to {index_path}")
    faiss.write_index(faiss_index, index_path)
    
    # Save metadata mapping
    metadata_mapping = {
        i: {
            'source': doc.metadata['source'],
            'chunk': doc.metadata['chunk'],
            'chunk_total': doc.metadata.get('chunk_total', 1)
        } for i, doc in enumerate(processed_chunks)
    }
    
    metadata_path = os.path.join(args.output, "metadata_mapping.json")
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_mapping, f, indent=2)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()