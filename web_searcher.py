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
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import tempfile
import shutil

api_key = os.getenv("GEMINI_API_KEY_2")
if not api_key:
    raise ValueError("❌ Falta la variable de entorno: GEMINI_API_KEY_2")
genai.configure(api_key=api_key)


def create_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--remote-debugging-port=9222")

    # Crear un directorio temporal para los datos de usuario
    temp_user_data_dir = tempfile.mkdtemp()
    chrome_options.add_argument(f"--user-data-dir={temp_user_data_dir}")

    # Usar webdriver-manager para manejar la descarga de ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.temp_user_data_dir = temp_user_data_dir
    return driver
    except WebDriverException as e:
        raise RuntimeError(f"❌ Error inicializando ChromeDriver: {e}")

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

    try:
        driver.quit()
    finally:
        if hasattr(driver, "temp_user_data_dir"):
            shutil.rmtree(driver.temp_user_data_dir, ignore_errors=True)

    return results

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
