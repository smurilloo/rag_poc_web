import os
import tempfile
from azure.storage.blob import ContainerClient
from PyPDF2 import PdfReader
from vectorizacion import index_pdf_chunks  # Para indexar solo nuevos PDFs

# Configuración desde variables de entorno
container_url = os.getenv("AZURE_STORAGE_SAS_TOKEN")
if not container_url:
    raise ValueError("❌ Falta AZURE_STORAGE_SAS_TOKEN")

container_client = ContainerClient.from_container_url(f"{container_url}")

# Cache local para PDFs ya procesados
PDF_CACHE = {}

def load_pdfs_azure():
    pdfs = []
    metadatas = []

    try:
        blobs = list(container_client.list_blobs(name_starts_with="BD_Knowledge"))
    except Exception as e:
        raise RuntimeError(f"❌ Error al listar blobs en BD_Knowledge: {e}")

    new_pdfs_to_index = []

    for blob in blobs:
        if not blob.name.endswith(".pdf"):
            continue

        # Si ya está en cache, usarlo
        if blob.name in PDF_CACHE:
            pdf_data, metadata = PDF_CACHE[blob.name]
            pdfs.append(pdf_data)
            metadatas.append(metadata)
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
                pdf_data = {
                    "filename": filename,
                    "title": title,
                    "pages_texts": pages_texts
                }
                metadata = {
                    "filename": filename,
                    "title": title,
                    "pages": compress_page_ranges(pages_numbers)
                }

                # Guardar en cache
                PDF_CACHE[blob.name] = (pdf_data, metadata)

                pdfs.append(pdf_data)
                metadatas.append(metadata)

                # Añadir a la lista de nuevos PDFs a indexar
                new_pdfs_to_index.append(pdf_data)

        except Exception as e:
            print(f"⚠️ Error procesando PDF {blob.name}: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Solo indexar PDFs nuevos en Qdrant
    if new_pdfs_to_index:
        index_pdf_chunks(new_pdfs_to_index)

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
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = p
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


# Otros proveedores de nube
def load_pdfs_google():
    return [], []

def load_pdfs_aws():
    return [], []
