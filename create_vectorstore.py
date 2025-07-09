from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import streamlit as st
# initialize vectorstore_path
vectorestore_path = "vectordatabase/faiss_index"

def save_vectorDatabase(vectorstore):
    """
    Save the vectorstore to a specified path.
    """
   # Ensure base directory exists
    base_dir = os.path.dirname(vectorestore_path)
    os.makedirs(base_dir, exist_ok=True)

    # If the exact path exists and is a file, remove it
    if os.path.exists(vectorestore_path) and not os.path.isdir(vectorestore_path):
        os.remove(vectorestore_path)

    vectorstore.save_local(vectorestore_path) 
    # os.makedirs(vectorestore_path,exist_ok=True)
    # vectorstore.save_local(vectorestore_path)

def load_vectorDatabase():
    """
    load the vectorstore from a specified path.
    """
    if os.path.exists(vectorestore_path):
        embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(vectorestore_path, embedding,allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"Vectorstore not found at {vectorestore_path}")

@st.cache_resource
def process_add_and_pdfs_to_vectorstore(pdf_files,_existing_vectorstore=None):
    """
    Process a list of PDF files, extract text, split it into chunks, and add it to the vectorstore.
    """
    all_documents = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        # Load the PDF file
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()

        # text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
        documents = text_splitter.split_documents(pages)
        all_documents.extend(documents)

        # unlink the temporary file
        os.unlink(temp_file_path)
    
    # create embeddings
    vector_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if _existing_vectorstore:
        # If an existing vectorstore is provided, add new documents to it
        _existing_vectorstore.add_documents(all_documents)
        return _existing_vectorstore
    else:
        # Create a new vectorstore
        vectorstore = FAISS.from_documents(all_documents, vector_embedding)
        return vectorstore
