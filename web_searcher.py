# Este código busca artículos científicos en Google Scholar usando web Scrapping con Selenium,
# extrae títulos, resúmenes y enlaces, y luego genera un resumen claro y organizado
# con ayuda de un modelo desplegado en Azure AI Foundry.

from openai import AzureOpenAI
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import textwrap
import os

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
                results.append({"title": title, "url": url, "snippet": snippet})
            except Exception:
                continue
    driver.quit()
    return results

def get_annotated_summary(query: str) -> str:
    papers = get_web_papers_selenium(query)
    if not papers:
        return "No se encontraron artículos."

    prompt = "".join(
        f"Título: {p['title']}\nResumen: {p['snippet']}\nURL: {p['url']}\n\n" for p in papers
    )

    full_prompt = f"""
Analiza los siguientes artículos de Google Scholar y resume en máximo 4 párrafos:

{prompt}
"""

    response = client.chat.completions.create(
        deployment_id=deployment, 
        messages=[
            {"role": "system", "content": "Eres un asistente útil que resume papers académicos."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )

    raw_summary = response.choices[0].message.content if response and response.choices else "No se recibió respuesta."

    wrapped_summary = "\n".join(
        textwrap.fill(line, width=80) for line in raw_summary.splitlines()
    )
    return wrapped_summary


