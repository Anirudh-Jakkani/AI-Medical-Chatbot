from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


import os
from dotenv import load_dotenv

load_dotenv()

#Extract text from PDF files

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Filter documents 

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    given a list of Document objects, return a new list of Document objects
    containing only "source" in metadata and the origunal page_content.
    """
    minimal_docs : List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
           
        )
       
    return minimal_docs


# split the text into smaller chunks
def text_split(minimal_docs): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks

# download embeddings model and create vector store
def download_embeddings():
    """
    Download and return HuggingFace BGE embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        )
    
    return embeddings

