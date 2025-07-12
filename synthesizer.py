# Este código toma una pregunta y busca información en documentos PDF y artículos en internet,
# luego usa inteligencia artificial para combinar y resumir esa información en una respuesta clara y organizada.


import google.generativeai as genai
import textwrap
from vectorizacion import client, COLLECTION_NAME
from sentence_transformers import SentenceTransformer  # Importar SentenceTransformer

genai.configure(api_key="AIzaSyBnzr9P1NSCcF36lXHtf1tA5I9gfIiCcmg")

# Inicializar el modelo de SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def search_qdrant(query, top_k=5):
    # Codificar la consulta
    query_vector = encoder.encode(query).tolist()

    # Buscar en la colección de Qdrant
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    # Procesar los resultados
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

def synthesize_answer(query, pdfs, pdf_metadata, memory, web_papers):
    # Consultar la base de datos vectorial
    qdrant_results = search_qdrant(query, top_k=5)

    pdf_section = ""
    instruccion_archivos = ""
    documents = ""

    if pdfs and pdf_metadata:
        # Construir listado para fuentes - PDFs
        pdf_list_text = "\n".join(
            f"- {item['filename']} - {item['title']} (páginas: {item['pages']})"
            for item in pdf_metadata
        )
        pdf_section = f"Fuentes PDF consultadas:\n{pdf_list_text}\n"
        instruccion_archivos = (
            "IMPORTANTE: El modelo no tiene acceso a los archivos originales, "
            "solo al contenido textual proporcionado. Menciona explícitamente las fuentes citadas "
            "usando el formato 'nombre_archivo.pdf - Título del paper (páginas)'. "
            "Usa las páginas específicas donde aparece la información relevante."
        )

        # Concatenar texto por páginas con marca de página
        parts = []
        for pdf in pdfs:
            for page in pdf['pages_texts']:
                parts.append(f"[{pdf['filename']} - Página {page['page']}]\n{page['text']}")
        documents = "\n\n".join(parts)

    # Para papers web
    web_section = ""
    instruccion_web = ""
    if web_papers:
        # Dividir los textos web en páginas de 500 caracteres y agregar la numeración de página
        web_parts = []
        for wp in sorted(web_papers, key=lambda x: x.get("score", 0), reverse=True):
            title = wp['title']
            url = wp['url']
            snippet = wp['snippet']

            # Dividir el resumen en bloques de 500 caracteres (simulando las páginas)
            page_num = 1
            for i in range(0, len(snippet), 500):
                page_text = snippet[i:i+500]
                web_parts.append(f"{url} - {title} (página {page_num})\n{page_text}")
                page_num += 1

        web_section = f"Artículos web relevantes desde Google Scholar:\n" + "\n\n".join(web_parts) + "\n"

        instruccion_web = (
            "A partir de los artículos web anteriores, redacta un análisis claro y conciso, "
            "incorporando los siguientes elementos:\n"
            "- Usa títulos y URLs destacados en líneas propias.\n"
            "- Utiliza el formato 'url del paper - Título del paper (páginas)', "
            "indicando las páginas específicas donde aparece la información relevante.\n"
            "- Cada 500 caracteres del contenido se considera una página. "
            "Es decir, los primeros 500 caracteres corresponden a la página 1, "
            "los siguientes 500 a la página 2, y así sucesivamente.\n"
            "- Asegúrate de que el análisis mencione las páginas específicas y los fragmentos de contenido, "
            "siempre citando el número de página correspondiente.\n"
            "- Usa viñetas o numeración para temas comunes o puntos importantes.\n"
            "- Añade saltos de línea para mejorar la lectura.\n"
        )

    # Agregar resultados de Qdrant al prompt
    qdrant_section = ""
    if qdrant_results:
        qdrant_text = "\n\n".join(
            f"[{result['type']}] {result['filename'] if result['type'] == 'pdf' else result['url']} - {result['title']} (página {result['page']})\n{result['content']}"
            for result in qdrant_results
        )
        qdrant_section = f"Resultados de la base de datos vectorial:\n{qdrant_text}\n"

    prompt = f"""
Contexto previo:
{memory}

Consulta: {query}

Fuentes documentales (texto extraído de PDFs, segmentado por página):
{documents}

{pdf_section}
{instruccion_archivos}

{web_section}
{instruccion_web}

{qdrant_section}

Estructura la respuesta iniciando con los hallazgos de los PDFs y luego el análisis de los papers web.
Usa formato claro, con títulos y URLs respectivas de cada paper analizado, usa viñetas para puntos clave y saltos de línea adecuados.
El último parrafo debe ser un resumen sintetizando la información para responder de manera clara la pregunta realizada desde la interfaz web de manera concreta, sin ambiguedad.
"""

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    raw_summary = response.text.strip()

    wrapped_summary = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())

    return wrapped_summary