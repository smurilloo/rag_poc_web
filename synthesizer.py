import os
import json
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from vectorizacion import client, COLLECTION_NAME
from fastapi.responses import StreamingResponse 

# ===============================
# Configuración desde variables de entorno
# ===============================
api_key = os.getenv("OPEN_AI_API_KEY_1")
endpoint = os.getenv("OPEN_AI_ENDPOINT")
deployment = os.getenv("OPEN_AI_DEPLOYMENT") 

if not api_key or not endpoint or not deployment:
    raise ValueError("Faltan variables de entorno OPEN_AI_API_KEY_1, OPEN_AI_ENDPOINT o OPEN_AI_DEPLOYMENT")

client_aoai = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# Función: buscar en Qdrant
# ===============================
def search_qdrant(query, top_k=5):
    query_vector = encoder.encode(query).tolist()
    try:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
    except Exception as e:
        return [{"type": "error", "content": f"Error en búsqueda Qdrant: {str(e)}"}]

    results = []
    for hit in hits:
        payload = hit.payload
        source_field = None
        if payload.get("type") == "pdf":
            source_field = payload.get("filename", "")
        else:
            source_field = payload.get("url", "")

        results.append({
            "type": payload.get("type", "unknown"),
            "source": source_field,
            "title": payload.get("title", ""),
            "page": payload.get("page", 1),
            "score": hit.score,
            "content": payload.get("content", "")
        })
    return results

# ===============================
# Helper: dividir textos en chunks
# ===============================
def chunk_text(items, max_chars=2000):
    chunks = []
    current_chunk = ""
    for item in items:
        if 'pages_texts' in item:
            for page in item['pages_texts']:
                text = f"[{item['filename']} - Página {page['page']}]\n{page['text']}\n\n"
                if len(current_chunk) + len(text) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = text
                else:
                    current_chunk += text
        else:
            snippet = item.get("content", "")
            if not snippet:
                continue
            page_num = item.get("page", 1)
            source = item.get("source", "")
            title = item.get("title", "")
            text = f"[{source} - Página {page_num}] {title}\n{snippet}\n\n"
            if len(current_chunk) + len(text) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = text
            else:
                current_chunk += text
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ===============================
# Función: síntesis de respuesta segura con STREAMING
# ===============================
def synthesize_answer(query, pdfs, pdf_metadata, memory, web_papers):
    try:
        qdrant_results = search_qdrant(query, top_k=5)

        content_items = []
        if pdfs:
            content_items.extend(pdfs)
        if web_papers:
            content_items.extend(web_papers)
        if qdrant_results:
            content_items.extend(qdrant_results)

        text_chunks = chunk_text(content_items, max_chars=2000)
        memory_safe = memory[-2000:] if memory else ""

        def event_stream():
            for chunk in text_chunks:
                prompt = f"""
Contexto previo:
{memory_safe}

Consulta:
{query}

Información relevante:
{chunk}

Responde en máximo 4 párrafos. Cita fuentes y páginas donde corresponda.
"""
                if len(prompt) > 6000:
                    prompt = prompt[:6000]

                if not prompt.strip():
                    continue

                # Llamada en modo streaming
                with client_aoai.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": "Eres un asistente que resume PDFs y artículos académicos."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=400,
                    top_p=1.0,
                    stream=True
                ) as stream:
                    for event in stream:
                        if event.choices and event.choices[0].delta:
                            delta = event.choices[0].delta.get("content", "")
                            if delta:
                                yield delta

        # Enviar streaming a FastAPI
        return StreamingResponse(event_stream(), media_type="text/plain")

    except Exception as e:
        return StreamingResponse(
            iter([f"Error al generar respuesta: {str(e)}"]),
            media_type="text/plain"
        )

