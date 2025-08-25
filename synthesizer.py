import os
import textwrap
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
            "title": payload["title"],
            "page": payload["page"],
            "score": hit.score,
            "content": payload.get("content", "")
        })
    return results

# ===============================
# Helper: dividir textos en chunks
# ===============================
def chunk_text(items, max_chars=25000):
    """
    Divide contenido en bloques que no excedan max_chars caracteres (~25k tokens).
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
        else:  # Web papers
            snippet = item.get("snippet", "")
            for i in range(0, len(snippet), 500):
                page_text = snippet[i:i+500]
                page_num = i // 500 + 1
                text = f"{item.get('url')} - {item.get('title')} (página {page_num})\n{page_text}\n\n"
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
        qdrant_results = search_qdrant(query, top_k=1)

        # Construir listas de contenido para chunking
        content_items = []
        if pdfs:
            content_items.extend(pdfs)
        if web_papers:
            content_items.extend(web_papers)
        if qdrant_results:
            content_items.extend(qdrant_results)

        # Dividir en chunks de máximo 25k caracteres (~tokens)
        text_chunks = chunk_text(content_items, max_chars=25000)

        summaries = []
        for chunk in text_chunks:
            prompt = f"""
Contexto previo:
{memory}

Consulta:
{query}

Información relevante:
{chunk}

Responde en máximo 4 párrafos. Cita fuentes y páginas donde corresponda.
"""
            # Validar que prompt no esté vacío
            if not prompt.strip():
                continue

            response = client_aoai.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Eres un asistente que resume PDFs y artículos académicos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
                top_p=1.0
            )

            # 🔑 Validación robusta de la respuesta
            try:
                raw_summary = (
                    response.choices[0].message.content
                    if response and getattr(response.choices[0], "message", None)
                    else "No se recibió respuesta."
                )
                # Limpiar caracteres no JSON / invisibles
                raw_summary = raw_summary.replace("\x00", "").strip()
            except Exception:
                raw_summary = "No se recibió respuesta."

            wrapped_summary = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())
            summaries.append(wrapped_summary.strip())

        return "\n\n".join(summaries)

    except Exception as e:
        return f"Error al generar respuesta: {str(e)}"
