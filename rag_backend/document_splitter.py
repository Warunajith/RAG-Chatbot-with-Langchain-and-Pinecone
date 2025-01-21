from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunks(docs):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)

    splits = text_splitter.split_documents(docs)

    return splits