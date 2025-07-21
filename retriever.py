# Este código lee todos los archivos PDF en la carpeta del proyecto, extrae el texto de cada página,
# identifica los títulos y organiza la información para que pueda ser usada en respuestas o análisis.
# También resume las páginas leídas en rangos para mostrar de forma clara qué páginas se usaron.

import os
import tempfile
from azure.storage.blob import ContainerClient
from PyPDF2 import PdfReader

# Configuración desde variables de entorno
container_url = os.getenv("AZURE_STORAGE_SAS_TOKEN")
if not container_url:
    raise ValueError("❌ Falta AZURE_STORAGE_SAS_TOKEN")

# Inicializar cliente de contenedor directamente con la URL SAS completa
container_client = ContainerClient.from_container_url(f"{container_url}")

# ------------------------
# FUNCIONES DE CARGA DE PDF
# ------------------------

def load_pdfs_azure():
    pdfs = []
    metadatas = []

    try:
        blobs = container_client.list_blobs(name_starts_with="BD_Knowledge")
    except Exception as e:
        raise RuntimeError(f"❌ Error al listar blobs en BD_Knowledge: {e}")

    for blob in blobs:
        if not blob.name.endswith(".pdf"):
            continue

        print(f"📥 Descargando: {blob.name}")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                downloader = container_client.download_blob(blob.name)
                downloader.download_to_stream(temp_pdf)
                temp_pdf_path = temp_pdf.name
        except Exception as e:
            print(f"⚠️ Error al descargar {blob.name}: {e}")
            continue

        try:
            reader = PdfReader(temp_pdf_path)
            pages_texts = []
            pages_numbers = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_texts.append({"page": i + 1, "text": text.strip()})
                    pages_numbers.append(i + 1)

            if pages_texts:
                title = pages_texts[0]["text"].split("\n")[0].strip()
                filename = os.path.basename(blob.name)

                pdfs.append({
                    "filename": filename,
                    "title": title,
                    "pages_texts": pages_texts
                })

                metadatas.append({
                    "filename": filename,
                    "title": title,
                    "pages": compress_page_ranges(pages_numbers)
                })

        except Exception as e:
            print(f"⚠️ Error procesando PDF {blob.name}: {e}")

        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    return pdfs, metadatas

def compress_page_ranges(pages):
    if not pages:
        return ""
    pages = sorted(set(pages))
    ranges = []
    start = prev = pages[0]

    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = prev = p

    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)

# Otros proveedores de nube
def load_pdfs_google():
    return [], []

def load_pdfs_aws():
    return [], []
