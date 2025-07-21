# Este código busca artículos científicos en Google Scholar usando web Scrapping con Selenium,
# extrae títulos, resúmenes y enlaces, y luego genera un resumen claro y organizado
# con ayuda de una inteligencia artificial para facilitar la comprensión del contenido.

import google.generativeai as genai
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import textwrap
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# -----------------------------
# Configuración desde variables de entorno
# -----------------------------
api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta GEMINI_API_KEY_2")

genai.configure(api_key=api_key)


# Ruta al chromedriver instalado manualmente cuando se usa Dockerfile
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

def get_web_papers_selenium(query: str, max_pages: int = 10) -> List[Dict]:
    base_url = "https://scholar.google.com/scholar"

    # Opciones para entorno Docker
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--remote-debugging-port=9222")

    # Inicia el navegador con el chromedriver ubicado en el sistema
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)

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
Analiza los siguientes artículos científicos obtenidos de Google Scholar y genera un resumen claro y estructurado en formato tipo documento. Por favor:

- Usa dos párrafos separados.
- Incorpora títulos y URLs destacados en líneas propias.
  usando el formato 'url paper - Título del paper (páginas)',
  Usa las páginas específicas donde aparece la información relevante,
  ten en cuenta que cada 500 caracteres se pasa de una página a otra,
  es decir, los primeros 500 caracteres son la página 1, a los 1000 es la página 2,
  y así sucesivamente.
- Usa viñetas o numeración para temas comunes o puntos importantes.
- Añade saltos de línea para facilitar la lectura.
- No dejes líneas con más de 80 caracteres; usa saltos de línea para ajustar el texto.

Aquí están los artículos a analizar:

{prompt}
"""

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(full_prompt)
    raw_summary = response.text.strip()

    # Aplicar wrap para evitar líneas muy largas, respetando saltos de línea
    wrapped_summary = "\n".join(
        textwrap.fill(line, width=80) for line in raw_summary.splitlines()
    )

    return wrapped_summary
