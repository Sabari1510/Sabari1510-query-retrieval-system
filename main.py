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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

rag_chain: Runnable = None
FAISS_INDEX_PATH = "faiss_index"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("Application startup: Loading RAG chain...")
    try:
        # Step 1: Initialize the lightweight Google embedding client
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Step 2: Load the pre-built FAISS index from disk
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
        
        # Step 3: Set up the fast Groq LLM for answer generation
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        
        # Step 4: Create the final RAG chain
        prompt_template = """
        You are a factual answering assistant. Your task is to answer the QUESTION based ONLY on the provided CONTEXT.
        Your final answer MUST be concise and limited to a maximum of 3 sentences. First, state the most direct answer possible, then briefly state the main condition. Do not add extra information.
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

app = FastAPI(
    title="Stable Hybrid Query System (Groq + Google)", 
    version="11.0.0", 
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