import os
import json
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from vectorizacion import client, COLLECTION_NAME

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
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    results = []
    for hit in hits:
        payload = hit.payload
        results.append({
            "type": payload["type"],
            "filename" if payload["type"] == "pdf" else "url": payload.get("filename", payload.get("url")),
            "title": payload.get("title", ""),
            "page": payload.get("page", 1),
            "score": hit.score,
            "content": payload.get("content", "")
        })
    return results

# ===============================
# Helper: dividir textos en chunks
# ===============================
def chunk_text(items, max_chars=2000):  # reducido para evitar exceso de tokens
    """
    Divide contenido en bloques que no excedan max_chars caracteres (~500 tokens aprox).
    """
    chunks = []
    current_chunk = ""
    for item in items:
        if 'pages_texts' in item:  # PDFs
            for page in item['pages_texts']:
                text = f"[{item['filename']} - Página {page['page']}]\n{page['text']}\n\n"
                if len(current_chunk) + len(text) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = text
                else:
                    current_chunk += text
        else:  # Web papers o Qdrant results
            snippet = item.get("content", "")
            if not snippet:
                continue
            page_num = item.get("page", 1)
            source = item.get("filename", item.get("url", ""))
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
# Función: síntesis de respuesta segura
# ===============================
def synthesize_answer(query, pdfs, pdf_metadata, memory, web_papers):
    try:
        qdrant_results = search_qdrant(query, top_k=5)

        # Construir lista de contenido para chunking
        content_items = []
        if pdfs:
            content_items.extend(pdfs)
        if web_papers:
            content_items.extend(web_papers)
        if qdrant_results:
            content_items.extend(qdrant_results)

        # Dividir en chunks más pequeños
        text_chunks = chunk_text(content_items, max_chars=2000)

        # Limitar memoria para no romper límite de tokens
        memory_safe = memory[-2000:] if memory else ""

        summaries = []
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

            # Si prompt excede 6000 caracteres, lo recortamos
            if len(prompt) > 6000:
                prompt = prompt[:6000]

            if not prompt.strip():
                continue

            response = client_aoai.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Eres un asistente que resume PDFs y artículos académicos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400,
                top_p=1.0
            )

            try:
                raw_summary = (
                    json.dumps(
                        response.choices[0].message.to_dict(),
                        ensure_ascii=False
                    )
                    if response and getattr(response.choices[0], "message", None)
                    else '{"content":"No se recibió respuesta.","role":"assistant"}'
                )
                raw_summary = raw_summary.replace("\x00", "").strip()
            except Exception:
                raw_summary = '{"content":"No se recibió respuesta.","role":"assistant"}'

            summaries.append(raw_summary)

        # Combinar todas las respuestas JSON en una lista
        return "[" + ",".join(summaries) + "]"

    except Exception as e:
        return json.dumps({"content": f"Error al generar respuesta: {str(e)}", "role": "assistant"})

