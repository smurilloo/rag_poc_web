# Esta aplicación web recibe preguntas, busca respuestas en documentos PDF y artículos en internet,
# y devuelve una respuesta clara usando la información encontrada, recordando conversaciones previas.

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from memory_keeper import MemoryKeeper
from retriever import load_pdfs_azure, load_pdfs_google, load_pdfs_aws
from synthesizer import synthesize_answer
from web_searcher import get_web_papers_selenium
from vectorizacion import (
    index_pdf_chunks,
    index_web_papers,
    get_first_k_points,
    cleanup_collection,
    delete_collection
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

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "").strip()
    source = data.get("source", "azure").lower().strip()

    if not question:
        return JSONResponse(content={"answer": "Por favor ingresa una consulta válida."}, status_code=400)

    try:
        # 1. Cargar PDFs según la fuente seleccionada
        if source == "azure":
            pdf_texts_by_pages, pdf_metadata = load_pdfs_azure()
        elif source == "google":
            pdf_texts_by_pages, pdf_metadata = load_pdfs_google()
        elif source == "aws":
            pdf_texts_by_pages, pdf_metadata = load_pdfs_aws()
        else:
            return JSONResponse(content={"answer": f"Fuente no reconocida: {source}"}, status_code=400)

        # 2. Web scraping de artículos
        web_papers = get_web_papers_selenium(question)

        # 3. Limpiar parcialmente la colección (mantener solo algunos si aplica)
        cleanup_collection(limit=20)

        # 4. Indexar nuevos datos
        index_pdf_chunks(pdf_texts_by_pages)
        index_web_papers(web_papers)

        # 5. Contexto de memoria
        memory = memory_keeper.get_context()

        # 6. Generar respuesta
        answer = synthesize_answer(question, pdf_texts_by_pages, pdf_metadata, memory, web_papers)

        # 7. Guardar en memoria la conversación
        memory_keeper.remember(question, answer)

        # 8. 🔥 Eliminar la colección completa SOLO después de haber respondido
        delete_collection()

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(
            content={"answer": f"Error procesando la consulta: {str(e)}"},
            status_code=500
        )

@app.get("/inspect", response_class=JSONResponse)
async def inspect():
    try:
        points = get_first_k_points(k=20)

        pdf_points = [p for p in points if p["payload"].get("type") == "pdf"]
        web_points = [p for p in points if p["payload"].get("type") == "web"]

        return JSONResponse(content={
            "pdf_points": pdf_points[:10],
            "web_points": web_points[:10]
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
