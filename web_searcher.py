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

import google.generativeai as genai

# ✅ Configura Gemini API key
api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta GEMINI_API_KEY_2")
genai.configure(api_key=api_key)

# ✅ Rutas fijas
CHROME_PATH = "/usr/bin/google-chrome"
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

# ✅ Instala Google Chrome desde script si no existe
def install_chrome():
    if not os.path.exists(CHROME_PATH):
        print("🔧 Instalando Google Chrome...")
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "wget", "gnupg2", "unzip", "apt-transport-https", "ca-certificates", "curl"], check=True)
        subprocess.run([
            "bash", "-c",
            "curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg"
        ], check=True)
        subprocess.run([
            "bash", "-c",
            'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list'
        ], check=True)
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "google-chrome-stable"], check=True)

# ✅ Instala Chromedriver desde script si no existe
def install_chromedriver():
    if not os.path.exists(CHROMEDRIVER_PATH):
        print("🔧 Instalando Chromedriver...")
        url = "https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.91/linux64/chromedriver-linux64.zip"
        subprocess.run(["wget", url, "-O", "chromedriver.zip"], check=True)
        subprocess.run(["unzip", "chromedriver.zip"], check=True)
        shutil.move("chromedriver-linux64/chromedriver", CHROMEDRIVER_PATH)
        os.chmod(CHROMEDRIVER_PATH, 0o755)
        shutil.rmtree("chromedriver-linux64")
        os.remove("chromedriver.zip")

# ✅ Inicializa entorno completo
install_chrome()
install_chromedriver()

# 🔍 Scraping con Selenium
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
Analiza los siguientes artículos científicos obtenidos de Google Scholar y genera un resumen claro y estructurado en formato tipo documento:

- Usa 4 párrafos separados.
- Incorpora títulos y URLs destacados en líneas propias,
  usando el formato 'url paper - Título del paper (páginas)'.
- Calcula la página suponiendo 500 caracteres por página.
- Usa viñetas o numeración para temas comunes o puntos importantes.
- Añade saltos de línea para facilitar la lectura.
- No dejes líneas con más de 80 caracteres.

Artículos a analizar:

{prompt}
"""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(full_prompt)
    raw_summary = response.text.strip()
    wrapped = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())
    return wrapped
