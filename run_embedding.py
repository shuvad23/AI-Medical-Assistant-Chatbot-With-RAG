import os
from load_store import create_or_update_chroma_vectorstore

PDF_FOLDER = "data/pdfs"

def main():
    # Step 1: Get all PDFs in the folder
    pdf_paths = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.endswith(".pdf")
    ]

    if not pdf_paths:
        print("❌ No PDFs found in data/pdfs/. Please place your files there.")
        return

    print(f"📄 Found {len(pdf_paths)} PDF(s):")
    for path in pdf_paths:
        print(f"   - {os.path.basename(path)}")

    print("⚙️ Building or updating Chroma vectorstore...")
    create_or_update_chroma_vectorstore(pdf_paths)
    print("✅ Vectorstore saved to disk at 'vectordatabase/chroma_db'")

if __name__ == "__main__":
    main()
