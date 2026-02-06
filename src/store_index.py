from dotenv import load_dotenv
import os

from src.helper_runtime import load_pdf_files, filter_to_minimal_docs, text_split
from src.helper_indexing import download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if __name__ == "__main__":
    print("Loading Pdfs...")
    extracted_data = load_pdf_files("data")
    
    print("Filtering Documents...")
    minimal_docs = filter_to_minimal_docs(extracted_data)
    
    print("Splitting Text into Chunks...")
    text_chunks = text_split(minimal_docs)
    
    
    print("Downloading Embeddings Model...")
    embeddings = download_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1")
        )
    else:
        print(" Pinecone index already exists")

    # 7. Upload vectors
    print("⬆️ Uploading embeddings to Pinecone...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print(" Pinecone indexing completed successfully!")