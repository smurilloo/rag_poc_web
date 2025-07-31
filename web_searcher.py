import os
import textwrap
import tempfile
import shutil
from typing import List, Dict

import google.generativeai as genai

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import subprocess

# === Setup Gemini ===
api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta la variable de entorno: GEMINI_API_KEY_2")
genai.configure(api_key=api_key)

# === Forzar instalación de ChromeDriver en ruta válida ===
def install_chromedriver_compatible_version(chrome_version_major: str = "124"):
    compatible_version = "124.0.6367.207"
    chromedriver_url = (
        f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/"
        f"{compatible_version}/linux64/chromedriver-linux64.zip"
    )
    dest_dir = "/home/site/wwwroot/bin"
    os.makedirs(dest_dir, exist_ok=True)

    try:
        print(f"✅ Descargando ChromeDriver compatible con Chrome {chrome_version_major}...")
        subprocess.run(
            ["wget", chromedriver_url, "-O", "chromedriver.zip"], check=True
        )
        subprocess.run(["unzip", "-o", "chromedriver.zip", "-d", "chromedriver_extracted"], check=True)
        chromedriver_src = "chromedriver_extracted/chromedriver-linux64/chromedriver"
        chromedriver_dest = os.path.join(dest_dir, "chromedriver")
        shutil.move(chromedriver_src, chromedriver_dest)
        os.chmod(chromedriver_dest, 0o755)
        print(f"✅ ChromeDriver instalado en {chromedriver_dest}")
    except Exception as e:
        print(f"⚠️ Error instalando ChromeDriver: {e}")

# === Inicializa ChromeDriver (headless) ===
def create_chrome_driver():
    install_chromedriver_compatible_version()

    chrome_bin_path = "/home/site/wwwroot/bin/google-chrome"
    chromedriver_path = "/home/site/wwwroot/bin/chromedriver"

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--remote-debugging-port=9222")

    temp_user_data_dir = tempfile.mkdtemp()
    chrome_options.add_argument(f"--user-data-dir={temp_user_data_dir}")

    if os.path.isfile(chrome_bin_path):
        chrome_options.binary_location = chrome_bin_path

    try:
        service = Service(executable_path=chromedriver_path)
        return webdriver.Chrome(service=service, options=chrome_options)
    except WebDriverException as e:
        raise RuntimeError(f"❌ Error inicializando ChromeDriver: {e}")

# 🔍 Scraping en Google Scholar
def get_web_papers_selenium(query: str, max_pages: int = 2) -> List[Dict]:
    driver = create_chrome_driver()
    driver.implicitly_wait(5)

    results = []
    for page in range(max_pages):
        start = page * 10
        search_url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}&start={start}"

        try:
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
        except Exception as e:
            print(f"⚠️ Error en página {page + 1}: {e}")
            continue

    driver.quit()
    return results

# ✍️ Resumen anotado con Gemini
def get_annotated_summary(query: str) -> str:
    papers = get_web_papers_selenium(query)
    if not papers:
        return "No se encontraron artículos científicos para esta consulta."

    prompt = "".join(
        f"Título: {p['title']}\nResumen: {p['snippet']}\nURL: {p['url']}\n\n" for p in papers
    )

    full_prompt = f"""
Analiza los siguientes artículos científicos de Google Scholar y genera un resumen
estructurado en formato documento:

- Usa 4 párrafos separados.
- En cada artículo, presenta el título y la URL en una línea como:
  'url del paper - Título del paper (páginas)'
- Calcula las páginas asumiendo 500 caracteres por página.
- Usa viñetas o numeración para agrupar temas similares.
- Añade saltos de línea frecuentes.
- No uses líneas de más de 80 caracteres.

Artículos:

{prompt}
"""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(full_prompt)
        raw_summary = response.text.strip()
        wrapped = "\n".join(textwrap.fill(line, width=80) for line in raw_summary.splitlines())
        return wrapped
    except Exception as e:
        return f"❌ Error al generar el resumen con Gemini: {e}"
