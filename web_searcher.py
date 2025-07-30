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

# ✅ Ruta fija para Chrome y Chromedriver en Azure App Service
CHROME_PATH = "/usr/bin/google-chrome"
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

# ✅ Instala Chrome si no existe
def ensure_chrome_installed():
    if not os.path.exists(CHROME_PATH):
        print("🔧 Instalando Google Chrome...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "wget", "gnupg2", "unzip", "apt-transport-https", "ca-certificates"], check=True)
        subprocess.run(["wget", "-q", "-O", "-", "https://dl.google.com/linux/linux_signing_key.pub"], stdout=subprocess.PIPE)
        subprocess.run(["sh", "-c", 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list'], check=True)
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "google-chrome-stable"], check=True)

# ✅ Instala Chromedriver si no existe
def ensure_chromedriver_installed():
    if not os.path.exists(CHROMEDRIVER_PATH):
        print("🔧 Instalando Chromedriver...")
        subprocess.run(["wget", "https://chromedriver.storage.googleapis.com/124.0.6367.91/chromedriver_linux64.zip", "-O", "chromedriver.zip"], check=True)
        subprocess.run(["unzip", "chromedriver.zip"], check=True)
        shutil.move("chromedriver", CHROMEDRIVER_PATH)
        os.chmod(CHROMEDRIVER_PATH, 0o755)

# ✅ Inicializa entorno
ensure_chrome_installed()
ensure_chromedriver_installed()

# 🔍 Scraping en Google Scholar
def get_web_papers_selenium(query: str, max_pages: int = 2) -> List[Dict]:
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
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
