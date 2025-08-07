import os
import requests
import uvicorn
from dotenv import load_dotenv

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

# --- In-memory cache for RAG chains ---
rag_chains_cache: Dict[str, Runnable] = {}

# --- Core Logic Function ---
def create_rag_chain_for_document(url: str) -> Runnable:
    """Processes a document and creates a RAG chain optimized for precise answers."""
    print(f"Processing new document. This will be slow. URL: {url}")
    temp_pdf_path = "temp_document.pdf"
    try:
        # Download and load the PDF
        response = requests.get(url)
        response.raise_for_status()
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)
        
        loader = PyMuPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings and the vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Configure the retriever to fetch the top 2 most relevant chunks
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
        
        # Configure the LLM
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        
        # A strict prompt to force short and precise answers
        prompt_template = """
        You are a factual answering assistant. Your task is to answer the QUESTION based ONLY on the provided CONTEXT.

        **Instructions:**
        1.  Find the direct answer to the QUESTION within the CONTEXT.
        2.  Your final answer MUST be concise and limited to a maximum of 3 sentences.
        3.  First, state the most direct answer possible (e.g., 'Yes, the policy covers this.', 'The waiting period is 36 months.').
        4.  Then, briefly state the most important condition or detail.
        5.  Do not add any extra information, greetings, or explanations.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION:
        {question}

        PRECISE ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the final RAG chain
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        print(f"Successfully created RAG chain for {url}")
        return rag_chain
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Precise Intelligent Query-Retrieval System",
    version="6.0.0"
)

# Pydantic models for the API
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Security dependency
security = HTTPBearer()
EXPECTED_TOKEN = "07ce76a034586438114a48d6ff4a5c6cabf5eaa94d7fb42920c62c795308f1d5"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authorization token")
    return credentials

# The main API endpoint
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)], tags=["Submissions"])
async def run_submission(request: HackRxRequest):
    doc_url = request.documents
    
    # Use the cache to avoid reprocessing documents
    if doc_url not in rag_chains_cache:
        try:
            # This slow step only runs once per document
            rag_chains_cache[doc_url] = create_rag_chain_for_document(doc_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not process new document: {str(e)}")
    
    rag_chain = rag_chains_cache[doc_url]
    
    # Process all questions quickly using the cached chain
    answers = []
    for question in request.questions:
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {question}")
            
    return HackRxResponse(answers=answers)

# This part allows you to run the app locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
