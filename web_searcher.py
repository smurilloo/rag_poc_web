from openai import AzureOpenAI
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import os
import json
from fastapi.responses import StreamingResponse 

# ---------------------------
# Configuración desde variables de entorno
# ---------------------------
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
# Funciones de búsqueda web con Selenium
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
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "page": page + 1
                })
            except Exception:
                continue
    driver.quit()
    return results

# ---------------------------
# División segura de texto
# ---------------------------
def chunk_text_for_tokens(items: List[Dict], max_chars: int = 2000) -> List[str]:
    """
    Divide los resultados en bloques pequeños (máx 2000 caracteres).
    """
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
# Resumen seguro con STREAMING
# ---------------------------
def get_annotated_summary(query: str):
    """
    Devuelve un StreamingResponse en tiempo real.
    """
    try:
        papers = get_web_papers_selenium(query)
        if not papers:
            return StreamingResponse(
                iter([json.dumps({"content": "No se encontraron artículos.", "role": "assistant"}, ensure_ascii=False)]),
                media_type="application/json"
            )

        text_chunks = chunk_text_for_tokens(papers, max_chars=2000)

        full_prompt = f"""
Analiza los siguientes artículos de Google Scholar y resume en máximo 3 párrafos:

{text_chunks[0]}
""".strip()

        def event_stream():
            with client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "Eres un asistente que resume papers académicos."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                top_p=1.0,
                stream=True  
            ) as stream:
                for event in stream:
                    if event.choices and event.choices[0].delta:
                        delta = event.choices[0].delta.get("content", "")
                        if delta:
                            yield delta

        return StreamingResponse(event_stream(), media_type="text/plain")

    except Exception as e:
        return StreamingResponse(
            iter([json.dumps({"content": f"Error procesando la búsqueda: {str(e)}", "role": "assistant"}, ensure_ascii=False)]),
            media_type="application/json"
        )
