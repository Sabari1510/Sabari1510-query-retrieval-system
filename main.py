import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List, Dict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request, HTTPException

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Global Variables ---
rag_chain_cache: Dict[str, Runnable] = {}
FAISS_INDEX_PATH = "faiss_index"

def get_rag_chain() -> Runnable:
    """Loads the pre-built RAG chain from the local FAISS index."""
    # This function is now called only when needed, not on startup.
    global rag_chain_cache
    
    # Use a simple key since we only have one index
    cache_key = "default_chain"

    if cache_key in rag_chain_cache:
        return rag_chain_cache[cache_key]

    print("First request: Loading pre-built FAISS index... This will be slow.")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
        
        prompt_template = """
        You are a factual answering assistant. Your task is to answer the QUESTION based ONLY on the provided CONTEXT.

        **Instructions:**
        1. Find the direct answer to the QUESTION within the CONTEXT.
        2. Your final answer MUST be concise and limited to a maximum of 3 sentences.
        3. First, state the most direct answer possible.
        4. Then, briefly state the most important condition or detail.
        5. Do not add any extra information.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION:
        {question}

        PRECISE ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        
        # Store the chain in the cache for future requests
        rag_chain_cache[cache_key] = rag_chain
        print("Loading complete. RAG chain is now cached.")
        return rag_chain

    except Exception as e:
        print(f"FATAL: Could not load RAG chain. Error: {e}")
        return None


# --- FastAPI Application Setup ---
# The lifespan manager has been REMOVED for stability.
app = FastAPI(
    title="Stable Query-Retrieval System",
    version="8.0.0"
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
    rag_chain = get_rag_chain()
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Service is not ready. RAG chain could not be loaded.")
    
    answers = []
    for question in request.questions:
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            
    return HackRxResponse(answers=answers)
