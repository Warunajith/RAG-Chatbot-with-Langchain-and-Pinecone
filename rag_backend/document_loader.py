from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import BSHTMLLoader



def load_text_file(filename:str):
   
    loader = TextLoader(filename)   # "/content/example_txt_file.txt"
    txt_data = loader.load()

    return txt_data


def load_text_files(filesDirectory:str):
    
    loader = DirectoryLoader(filesDirectory, glob="**/*.txt")
    txt_dataset = loader.load()

    return txt_dataset


def load_pdf_file(filename):

    loader = PyPDFLoader(filename)
    pdf_data = loader.load()

    return pdf_data


def load_pdf_files(filesDirectory:str):
    
    loader = PyPDFDirectoryLoader(filesDirectory)
    pdf_dataset = loader.load()

    return pdf_dataset


def load_csv_file(filename:str):
    
    loader = CSVLoader(filename)
    csv_data = loader.load()

    return csv_data


def load_html_file(filename:str):
    
    loader = BSHTMLLoader(filename)
    html_data = loader.load()

    return html_data