import os
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request, HTTPException

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

rag_chain: Runnable = None
FAISS_INDEX_PATH = "faiss_index"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup. It loads the pre-built
    search index and sets up the RAG chain.
    """
    global rag_chain
    print("Application startup: Loading RAG chain...")
    try:
        # Step 1: Use the fast, local HuggingFace model for embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Step 2: Load the pre-built FAISS index from disk
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
        
        # Step 3: Set up the Google Gemini LLM for answer generation
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        
        # Step 4: Create the final RAG chain with a high-accuracy prompt
        prompt_template = """
        You are a highly diligent and precise AI assistant for answering questions about an insurance policy.
        Your task is to answer the user's QUESTION based strictly on the provided CONTEXT sections.
        Synthesize a concise and direct answer. If the information is not available, state that clearly.

        CONTEXT:
        ---
        {context}
        ---
        QUESTION: {question}

        PRECISE ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        print("Pre-loading complete. RAG chain is ready.")
    except Exception as e:
        print(f"FATAL: Could not load RAG chain on startup. Error: {e}")
    yield
    print("Application shutdown.")

# --- FastAPI Application Setup ---
app = FastAPI(
    title="High-Accuracy Query System (Gemini)", 
    version="14.0.0", 
    lifespan=lifespan
)

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

# --- Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)], tags=["Submissions"])
async def run_submission(request: HackRxRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Service is not ready. RAG chain not loaded.")
    
    answers = []
    for question in request.questions:
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            
    return HackRxResponse(answers=answers)

# This part is for running the app locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)