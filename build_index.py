import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader # We are now using this loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# We will now use a local file path instead of a URL
LOCAL_PDF_PATH = "policy.pdf"
FAISS_INDEX_PATH = "faiss_index"

def build_index():
    print(f"Starting to build index from local file: {LOCAL_PDF_PATH}")
    
    # MODIFIED: Load directly from the local PDF file
    loader = UnstructuredPDFLoader(LOCAL_PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Document intelligently parsed and split into {len(splits)} chunks.")

    print("Creating embeddings with local HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index successfully built and saved to '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    build_index()