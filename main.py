
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from memory_keeper import MemoryKeeper
from retriever import load_pdfs_azure
from synthesizer import synthesize_answer
from web_searcher import get_web_papers_selenium
from vectorizacion import (
    index_web_papers,
    ensure_collection
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_keeper = MemoryKeeper()

@app.on_event("startup")
async def startup_event():
    ensure_collection()  

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/ask")
async def ask(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
    except Exception:
        return JSONResponse(
            content={"answer": "Error leyendo el request, envía un JSON válido."},
            status_code=400
        )

    if not question:
        return JSONResponse(
            content={"answer": "Por favor ingresa una consulta válida."},
            status_code=400
        )

    try:
        # Recuperar PDFs (solo descarga e indexa nuevos)
        pdf_texts_by_pages, pdf_metadata = load_pdfs_azure()

        # Buscar papers web
        web_papers = get_web_papers_selenium(question)

        # Indexar web papers (los PDFs ya se indexaron dentro de load_pdfs_azure)
        index_web_papers(web_papers)

        # Recuperar memoria contextual
        memory = memory_keeper.get_context()

        # Generar respuesta con Azure OpenAI
        answer = synthesize_answer(question, pdf_texts_by_pages, pdf_metadata, memory, web_papers)

        # Guardar en memoria
        memory_keeper.remember(question, answer)

        # Respuesta JSON serializable
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(content={"answer": f"Error procesando la consulta: {str(e)}"})
