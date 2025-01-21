from fastapi import FastAPI,Request,File, UploadFile,Form

import uvicorn
from logger import logging
from llm import get_llm,get_embedding_model
from langchain_core.output_parsers import StrOutputParser
from entity import ChatReq,SessionResponse
from prompt import get_prompt,get_contextualize_prompt
from langchain_core.messages import HumanMessage,AIMessage
from document_loader import load_pdf_file
from document_splitter import get_chunks

from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.document_loaders import PyPDFLoader
from uuid import uuid4
from io import BytesIO
import tempfile
import time
import os

app = FastAPI(
    title="Chatbot with Langchain",
    version="1.0",
    description="Simple Conversational RAG-Chatbot with OpenAI and Langchain"
)
logging.info("FastAPI app initialized" )



llm=get_llm()
logging.info("llm initialized")

embedding_model=get_embedding_model()
logging.info("embedding model initialized")


# Initialize the store for session histories
store = {}



@app.post("/api/rag-chatbot")
async def generate_chat(request:Request,file: UploadFile = File(...),question: str = Form(...),):

    # Retrieve session token from the Authorization header
    auth_header = request.headers.get("Authorization")

    session_token = None

    if auth_header and auth_header.startswith("Bearer "):
        session_token = auth_header.split("Bearer ")[1]
        logging.info(f"Session Token Recieved: {session_token}")
    else:
        logging.warning("Authorization header not found or invalid format")
 

    # Get the JSON part of the request
   
    question = question
    logging.info(f"Question: {question}")

     # Read the file content into memory (as bytes)
    file_content = await file.read()
    
    # Create a temporary file to save the in-memory file content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    # Load the PDF using PyPDFLoader from the temporary file
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    data_chunks=get_chunks(docs)
    logging.info("data chunks created and loaded")

    # Create a vector store from the document chunks

    # embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=pinecone_api_key)

    index_name = f"{session_token}"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:

        if len(existing_indexes)>=4:
            pc.delete_index(existing_indexes[0])
        
        pc.create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        index = pc.Index(index_name)

        vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

        uuids = [str(uuid4()) for _ in range(len(data_chunks))]

        vector_store.add_documents(documents=data_chunks, ids=uuids)

    logging.info("Vector store created with uploaded document in Pinecone")

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    retriever = vector_store.as_retriever(

        search_type="similarity"
        
    )
    



    contextualize_prompt= get_contextualize_prompt()

    #Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )
    logging.info("History Aware Retriver Created")

    prompt=get_prompt()

    # Create the question-answering chain
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # Create the history aware RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Create the conversational RAG chain with session history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    logging.info("Coversational RAG chain created")


    response = conversational_rag_chain.invoke(

        {"input": question},

        config={"configurable": {"session_id": session_token}},
    )

    

    answer=response["answer"]

    logging.info(f"Coversational RAG chain invoked and response is {answer}")

    return {"status": "success", "response": answer}



@app.post("/api/generate-session")
async def generate_session():
    session_token = str(uuid4())  # Generate a unique session token
    return SessionResponse(session_token=session_token)


@app.get("/")
async def home():
    response_data={
        "status": "success",
        "response":"Welcome to the Chatbot with Langchain"
    }

    return response_data



# Function to get the session history for a given session ID
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]





# if __name__ == "__main__":  

#     uvicorn.run(app,host="localhost",port=5000)