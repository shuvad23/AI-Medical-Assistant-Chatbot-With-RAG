from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# load raw pdf files -----
DATA_PATH = "data/"
def load_text_from_pdf_files(source):
    loader = DirectoryLoader(source,glob ='*.pdf',loader_cls =PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_text_from_pdf_files(source = DATA_PATH)


# create chunks 
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)


# create vector embeddings using Huggingface embedding models - "sentence-transformers/all-MiniLM-L6-v2"
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# store vector embeddings in FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
Database_FAISS_PATH = "vectorembedding/db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(Database_FAISS_PATH)