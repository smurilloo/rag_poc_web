# Este código busca artículos científicos en Google Scholar usando web Scrapping con Selenium,
# extrae títulos, resúmenes y enlaces, y luego genera un resumen claro y organizado
# con ayuda de una inteligencia artificial para facilitar la comprensión del contenido.

import os
import shutil
import subprocess
import textwrap
import time
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import google.generativeai as genai

# ✅ Configura Gemini API key
api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta GEMINI_API_KEY_2")
genai.configure(api_key=api_key)

# ✅ Ruta de instalación de Chrome y Chromedriver
CHROME_PATH = "/usr/local/bin/google-chrome"
CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver"

# ✅ Instala Chrome si no existe
def ensure_chrome_installed():
    if not os.path.exists(CHROME_PATH):
        print("🔧 Descargando Google Chrome...")
        subprocess.run([
            "wget", "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb", "-O", "chrome.deb"
        ])
        subprocess.run(["apt-get", "update"])
        subprocess.run(["apt-get", "install", "-y", "./chrome.deb"])
        shutil.move("/usr/bin/google-chrome", CHROME_PATH)

# ✅ Instala Chromedriver si no existe
def ensure_chromedriver_installed():
    if not os.path.exists(CHROMEDRIVER_PATH):
        print("🔧 Descargando Chromedriver...")
        subprocess.run([
            "wget", "https://chromedriver.storage.googleapis.com/124.0.6367.91/chromedriver_linux64.zip", "-O", "chromedriver.zip"
        ])
        subprocess.run(["apt-get", "install", "-y", "unzip"])
        subprocess.run(["unzip", "chromedriver.zip"])
        shutil.move("chromedriver", CHROMEDRIVER_PATH)
        os.chmod(CHROMEDRIVER_PATH, 0o755)

# ✅ Llama estas funciones al inicio
ensure_chrome_installed()
ensure_chromedriver_installed()

# 🔍 Scraping en Google Scholar
def get_web_papers_selenium(query: str, max_pages: int = 2) -> List[Dict]:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = CHROME_PATH

    service = Service(CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(5)

    results = []
    for page in range(max_pages):
        start = page * 10
        search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&start={start}"
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


# ✍️ Resumen usando Gemini
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

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(full_prompt)
    raw_summary = response.text.strip()
    wrapped_summary = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())
    return wrapped_summary
