import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    llm=ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key

    )

    return llm

def get_embedding_model():

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    return embedding_model