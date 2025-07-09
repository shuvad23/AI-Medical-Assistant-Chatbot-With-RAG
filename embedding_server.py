from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, tempfile, os
from load_store import process_pdf_to_chroma

app = FastAPI()

@app.post("/embed/")
async def embed_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            pdf_path = tmp.name

        process_pdf_to_chroma(pdf_path)
        os.remove(pdf_path)

        return JSONResponse({"status": "✅ success", "file": file.filename})
    except Exception as e:
        return JSONResponse({"status": "❌ failed", "error": str(e)}, status_code=500)


