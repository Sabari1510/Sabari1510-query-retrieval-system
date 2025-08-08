import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

KNOWN_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
FAISS_INDEX_PATH = "faiss_index"

def build_index():
    print("Starting to build index using Google Embeddings...")
    response = requests.get(KNOWN_DOCUMENT_URL)
    with open("temp.pdf", "wb") as f: f.write(response.content)
    
    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    os.remove("temp.pdf")

    print("Creating embeddings with Google's API...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index successfully built and saved to '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    build_index()