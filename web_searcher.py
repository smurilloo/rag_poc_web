from openai import AzureOpenAI
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import textwrap
import os
import json

# Configuración desde variables de entorno
api_key = os.getenv("OPEN_AI_API_KEY_1")
endpoint = os.getenv("OPEN_AI_ENDPOINT")
deployment = os.getenv("OPEN_AI_DEPLOYMENT")

if not api_key or not endpoint or not deployment:
    raise ValueError("Faltan OPEN_AI_API_KEY_1, OPEN_AI_ENDPOINT o OPEN_AI_DEPLOYMENT")

# Cliente Azure OpenAI
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint
)

# ---------------------------
# Funciones de búsqueda web
# ---------------------------
def get_web_papers_selenium(query: str, max_pages: int = 2) -> List[Dict]:
    base_url = "https://scholar.google.com/scholar"
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    driver = webdriver.Chrome(options=options)

    results = []
    for page in range(max_pages):
        start = page * 10
        search_url = f"{base_url}?q={query.replace(' ', '+')}&start={start}"
        driver.get(search_url)
        time.sleep(3)

        articles = driver.find_elements(By.CSS_SELECTOR, "div.gs_ri")
        for art in articles:
            try:
                title_elem = art.find_element(By.CSS_SELECTOR, "h3 a")
                title = title_elem.text.strip()
                url = title_elem.get_attribute("href")
                snippet_elem = art.find_elements(By.CLASS_NAME, "gs_rs")
                snippet = snippet_elem[0].text.strip() if snippet_elem else "No hay resumen disponible."
                results.append({"title": title, "url": url, "snippet": snippet, "page": 1})
            except Exception:
                continue
    driver.quit()
    return results

# ---------------------------
# Función para resumir con límite de tokens
# ---------------------------
def chunk_text_for_tokens(items: List[Dict], max_chars: int = 5000) -> List[str]:
    chunks = []
    current_chunk = ""
    for item in items:
        snippet_text = f"Título: {item['title']}\nResumen: {item.get('snippet', item.get('content',''))}\nURL: {item['url']}\n\n"
        if len(current_chunk) + len(snippet_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = snippet_text
        else:
            current_chunk += snippet_text
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ---------------------------
# Función de resumen final
# ---------------------------
def get_annotated_summary(query: str) -> str:
    """
    Devuelve SIEMPRE un string JSON con la forma:
    {"content": "<texto>", "role": "assistant"}
    para evitar errores de parseo.
    """
    try:
        papers = get_web_papers_selenium(query)
        if not papers:
            return json.dumps({"content": "No se encontraron artículos.", "role": "assistant"}, ensure_ascii=False)

        text_chunks = chunk_text_for_tokens(papers, max_chars=5000)
        # Usamos solo el primer chunk para construir un ÚNICO mensaje y retornarlo en JSON.
        full_prompt = f"""
Analiza los siguientes artículos de Google Scholar y resume en máximo 4 párrafos:

{text_chunks[0]}
""".strip()

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "Eres un asistente que resume papers académicos."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            top_p=1.0
        )

        # Extraemos SOLO el mensaje del asistente
        msg = getattr(response.choices[0], "message", None)
        content = (msg.content if msg and getattr(msg, "content", None) else "No se recibió respuesta.").replace("\x00", "").strip()
        role = (msg.role if msg and getattr(msg, "role", None) else "assistant")

        # Retornamos EXCLUSIVAMENTE JSON (sin texto adicional)
        return json.dumps({"content": content, "role": role}, ensure_ascii=False)

    except Exception:
        # En cualquier error, seguimos devolviendo JSON válido
        return json.dumps({"content": "No se recibió respuesta.", "role": "assistant"}, ensure_ascii=False)
