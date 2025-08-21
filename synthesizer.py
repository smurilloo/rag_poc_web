# Este código toma una pregunta y busca información en documentos PDF y artículos en internet,
# luego usa inteligencia artificial (Azure OpenAI desplegado en AI Foundry)
# para combinar y resumir esa información en una respuesta clara y organizada.

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

if not api_key or not endpoint:
    raise ValueError("❌ Falta OPEN_AI_API_KEY_1 o OPEN_AI_ENDPOINT en variables de entorno")

# Inicializar cliente Azure OpenAI
client_aoai = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
)

# Inicializar modelo de embeddings
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
# Función: síntesis de respuesta con Azure OpenAI
# ===============================
def synthesize_answer(query, pdfs, pdf_metadata, memory, web_papers):
    qdrant_results = search_qdrant(query, top_k=5)

    pdf_section = ""
    instruccion_archivos = ""
    documents = ""

    if pdfs and pdf_metadata:
        pdf_list_text = "\n".join(
            f"- {item['filename']} - {item['title']} (páginas: {item['pages']})"
            for item in pdf_metadata
        )
        pdf_section = f"Fuentes PDF consultadas:\n{pdf_list_text}\n"
        instruccion_archivos = (
            "Responde la pregunta realizada usando una respuesta corta de máximo 4 párrafos. "
            "IMPORTANTE: El modelo no tiene acceso a los archivos originales, "
            "solo al contenido textual proporcionado. Menciona explícitamente las fuentes citadas "
            "usando el formato 'nombre_archivo.pdf - Título del paper (páginas)'. "
            "Usa las páginas específicas donde aparece la información relevante."
        )

        parts = []
        for pdf in pdfs:
            for page in pdf['pages_texts']:
                parts.append(f"[{pdf['filename']} - Página {page['page']}]\n{page['text']}")
        documents = "\n\n".join(parts)

    web_section = ""
    instruccion_web = ""
    if web_papers:
        web_parts = []
        for wp in sorted(web_papers, key=lambda x: x.get("score", 0), reverse=True):
            title = wp['title']
            url = wp['url']
            snippet = wp['snippet']

            page_num = 1
            for i in range(0, len(snippet), 500):
                page_text = snippet[i:i+500]
                web_parts.append(f"{url} - {title} (página {page_num})\n{page_text}")
                page_num += 1

        web_section = f"Artículos web relevantes desde Google Scholar:\n" + "\n\n".join(web_parts) + "\n"

        instruccion_web = (
            "Responde la pregunta usando máximo 4 párrafos, a partir de los artículos web. "
            "Para cada fuente, comienza indicando 'url - Título (páginas)', "
            "y luego el análisis a partir de las páginas útiles. "
            "Indica las páginas específicas (cada 500 caracteres = 1 página). "
            "Puedes usar citas textuales o parafrasear, pero siempre citando el número de página. "
            "Usa viñetas o numeración para los puntos clave."
        )

    qdrant_section = ""
    if qdrant_results:
        qdrant_text = "\n\n".join(
            f"[{result['type']}] {result['filename'] if result['type'] == 'pdf' else result['url']} "
            f"- {result['title']} (página {result['page']})\n{result['content']}"
            for result in qdrant_results
        )
        qdrant_section = f"Resultados de la base de datos vectorial:\n{qdrant_text}\n"

    # Construir prompt
    prompt = f"""
Contexto previo:
{memory}

Consulta: {query}

Fuentes documentales (texto extraído de PDFs):
{documents}

{pdf_section}
{instruccion_archivos}

{web_section}
{instruccion_web}

{qdrant_section}

Estructura la respuesta iniciando con los hallazgos de los PDFs
y luego el análisis de los papers web.
Usa formato claro, con títulos, URLs, viñetas y saltos de línea.
"""

    # Llamar al modelo desplegado en Azure AI Foundry
    response = client_aoai.chat.completions.create(
        model="samue-mekqilbh-eastus_project",
        messages=[
            {"role": "system", "content": "Eres un asistente especializado en resumir información de PDFs y artículos académicos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1200,
        temperature=0.7,
    )

    raw_summary = response.choices[0].message.content.strip()
    wrapped_summary = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())
    return wrapped_summary


    return wrapped_summary

