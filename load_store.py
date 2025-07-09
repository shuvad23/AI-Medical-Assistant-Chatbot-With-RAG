from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

PDF_FOLDER = "data/pdfs"
VECTORSTORE_PATH = "vectordatabase/faiss_index2"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)

def create_or_update_vectorstore_from_files(pdf_paths, existing_vectorstore=None):
    """
    Load PDFs, split, embed, and create/update FAISS vectorstore.
    """
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(pages)
        all_docs.extend(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if existing_vectorstore:
        existing_vectorstore.add_documents(all_docs)
        existing_vectorstore.save_local(VECTORSTORE_PATH)
        return existing_vectorstore
    else:
        vs = FAISS.from_documents(all_docs, embeddings)
        vs.save_local(VECTORSTORE_PATH)
        return vs

def load_vectorDatabase2():
    if os.path.exists(VECTORSTORE_PATH):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
    return None



PDF_FOLDER = "data/pdfs"

# Step 1: Get all local PDFs
pdf_paths = [
    os.path.join(PDF_FOLDER, f)
    for f in os.listdir(PDF_FOLDER)
    if f.endswith(".pdf")
]

if not pdf_paths:
    print("❌ No PDFs found in data/pdfs/. Please place your files there.")
else:
    print(f"📄 Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(" -", p)

    print("⚙️ Building or updating vectorstore...")
    vectorstore = create_or_update_vectorstore_from_files(pdf_paths)
    print("✅ Vectorstore saved to disk.")
