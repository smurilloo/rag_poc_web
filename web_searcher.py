# Este código busca artículos científicos en Google Scholar usando web Scrapping con Selenium,
# extrae títulos, resúmenes y enlaces, y luego genera un resumen claro y organizado
# con ayuda de una inteligencia artificial para facilitar la comprensión del contenido.

import google.generativeai as genai
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import textwrap
import os
import shutil
import subprocess

# Configuración de la API Key de Gemini
api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta GEMINI_API_KEY_2")

genai.configure(api_key=api_key)

# Ruta de Chrome/Chromium esperada en App Service
CHROME_PATH = shutil.which("chromium-browser") or shutil.which("google-chrome") or "/usr/bin/google-chrome"
CHROMEDRIVER_PATH = shutil.which("chromedriver") or "/usr/local/bin/chromedriver"

# URL base de Google Scholar
base_url = "https://scholar.google.com/scholar"

def get_web_papers_selenium(query: str, max_pages: int = 2) -> List[Dict]:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.binary_location = CHROME_PATH

    # Validar existencia de Chrome y chromedriver
    if not os.path.exists(CHROME_PATH):
        raise EnvironmentError(f"Chrome no encontrado en {CHROME_PATH}")
    if not os.path.exists(CHROMEDRIVER_PATH):
        raise EnvironmentError(f"Chromedriver no encontrado en {CHROMEDRIVER_PATH}")

    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, options=chrome_options)
    driver.implicitly_wait(5)

    results = []
    for page in range(max_pages):
        start = page * 10
        search_url = f"{base_url}?q={query.replace(' ', '+')}&start={start}"
        driver.get(search_url)

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
"Usando una respuesta corta de máximo 4 párrafos realiza lo siguiente,"
Analiza los siguientes artículos científicos obtenidos de Google Scholar y genera un resumen claro y estructurado en formato tipo documento:
- Usa 4 párrafos separados.
- Incorpora títulos y URLs destacados en líneas propias.
- Usa viñetas o numeración para puntos clave.
- Añade saltos de línea para facilitar la lectura.
- No dejes líneas con más de 80 caracteres.

Aquí están los artículos a analizar:

{prompt}
"""

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(full_prompt)
    raw_summary = response.text.strip()

    wrapped_summary = "\n".join(
        textwrap.fill(line, width=80) for line in raw_summary.splitlines()
    )
    return wrapped_summary
