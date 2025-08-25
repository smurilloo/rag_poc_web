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
deployment = os.getenv("OPEN_AI_DEPLOYMENT")  # Ej: "gpt-35-turbo"

if not api_key or not endpoint or not deployment:
    raise ValueError("Faltan variables de entorno OPEN_AI_API_KEY_1, OPEN_AI_ENDPOINT o OPEN_AI_DEPLOYMENT")

# Inicializar cliente Azure OpenAI
client_aoai = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
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

    pdf_section, instruccion_archivos, documents = "", "", ""
    if pdfs and pdf_metadata:
        pdf_list_text = "\n".join(
            f"- {item['filename']} - {item['title']} (páginas: {item['pages']})"
            for item in pdf_metadata
        )
        pdf_section = f"Fuentes PDF consultadas:\n{pdf_list_text}\n"
        instruccion_archivos = (
            "Responde en máximo 4 párrafos. "
            "Menciona explícitamente las fuentes citadas "
            "usando el formato 'nombre_archivo.pdf - Título (páginas)'. "
        )

        parts = []
        for pdf in pdfs:
            for page in pdf['pages_texts']:
                parts.append(f"[{pdf['filename']} - Página {page['page']}]\n{page['text']}")
        documents = "\n\n".join(parts)

    web_section, instruccion_web = "", ""
    if web_papers:
        web_parts = []
        for wp in sorted(web_papers, key=lambda x: x.get("score", 0), reverse=True):
            title, url, snippet = wp['title'], wp['url'], wp['snippet']
            page_num = 1
            for i in range(0, len(snippet), 500):
                page_text = snippet[i:i+500]
                web_parts.append(f"{url} - {title} (página {page_num})\n{page_text}")
                page_num += 1
        web_section = "Artículos web relevantes:\n" + "\n\n".join(web_parts) + "\n"
        instruccion_web = (
            "Responde en máximo 4 párrafos, citando URL y título. "
            "Indica páginas (cada 500 caracteres = 1 página)."
        )

    qdrant_section = ""
    if qdrant_results:
        qdrant_text = "\n\n".join(
            f"[{r['type']}] {r['filename'] if r['type'] == 'pdf' else r['url']} "
            f"- {r['title']} (página {r['page']})\n{r['content']}"
            for r in qdrant_results
        )
        qdrant_section = f"Resultados vectoriales:\n{qdrant_text}\n"

    # Construir prompt
    prompt = f"""
Contexto previo:
{memory}

Consulta: {query}

Fuentes PDF:
{documents}

{pdf_section}
{instruccion_archivos}

{web_section}
{instruccion_web}

{qdrant_section}
"""

    # Llamada a Azure OpenAI con el deployment configurado
    response = client_aoai.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Eres un asistente que resume PDFs y artículos académicos."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=1.0
    )

    raw_summary = response.choices[0].message.content if response and response.choices else "No se recibió respuesta."

    wrapped_summary = "\n".join(
        textwrap.fill(line, width=80) for line in raw_summary.splitlines()
    )
    return wrapped_summary
