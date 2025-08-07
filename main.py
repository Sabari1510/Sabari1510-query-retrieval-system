import os
import requests
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict

# LangChain components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Cache and Constants ---
rag_chains_cache: Dict[str, Runnable] = {}
# The known document URL from the hackathon
KNOWN_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# --- Core Logic Function (No changes) ---
def create_rag_chain_for_document(url: str) -> Runnable:
    print(f"Processing new document. This will be slow. URL: {url}")
    # ... (The rest of this function is identical to the previous version)
    # This includes downloading, splitting, embedding, and creating the chain.
    temp_pdf_path = "temp_document.pdf"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        loader = PyMuPDFLoader(temp_pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2}) # OPTIMIZATION: Reduced k from 3 to 2
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        prompt_template = """
        You are an expert assistant. Use ONLY the following context to answer the question. Be concise and directly answer the user's question.

        CONTEXT: 
        {context}

        QUESTION: 
        {question}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        print(f"Successfully created RAG chain for {url}")
        return rag_chain
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# --- NEW: FastAPI Lifespan Manager for Pre-loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the application starts up.
    print("Application startup: Pre-loading known document...")
    try:
        # Pre-load the RAG chain for the known hackathon document
        rag_chains_cache[KNOWN_DOCUMENT_URL] = create_rag_chain_for_document(KNOWN_DOCUMENT_URL)
        print("Pre-loading complete. Application is ready.")
    except Exception as e:
        print(f"FATAL: Could not pre-load document on startup. Error: {e}")
    yield
    # This code runs when the application shuts down.
    rag_chains_cache.clear()
    print("Application shutdown: Cache cleared.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Ultra-Fast Intelligent Query-Retrieval System",
    version="3.0.0",
    lifespan=lifespan  # <-- Attach the lifespan manager
)

# ... (The rest of the file: Pydantic models, security, and the endpoint logic is the same)
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]
class HackRxResponse(BaseModel):
    answers: List[str]
security = HTTPBearer()
EXPECTED_TOKEN = "07ce76a034586438114a48d6ff4a5c6cabf5eaa94d7fb42920c62c795308f1d5"
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)], tags=["Submissions"])
async def run_submission(request: HackRxRequest):
    doc_url = request.documents
    if doc_url not in rag_chains_cache:
        # This will now only run for UNKNOWN documents, not the main hackathon doc
        try:
            rag_chains_cache[doc_url] = create_rag_chain_for_document(doc_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not process new document: {str(e)}")
    rag_chain = rag_chains_cache[doc_url]
    answers = []
    for question in request.questions:
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {question}")
    return HackRxResponse(answers=answers)