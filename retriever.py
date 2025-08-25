import os
import tempfile
from azure.storage.blob import ContainerClient
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from vectorizacion import index_pdf_chunks

# ===============================
# Variables de entorno
# ===============================
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1", "").strip().strip('"')
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')
QDRANT_COLLECTION = "vector_bd"

container_url = os.getenv("AZURE_STORAGE_SAS_TOKEN")
if not container_url:
    raise ValueError("❌ Falta AZURE_STORAGE_SAS_TOKEN")

# ===============================
# Inicialización de clientes
# ===============================
container_client = ContainerClient.from_container_url(f"{container_url}")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ===============================
# Utilidades
# ===============================
def get_indexed_filenames():
    """
    Consulta a Qdrant para obtener los filenames ya cargados en la colección.
    Retorna un set con los nombres.
    """
    try:
        indexed = set()
        scroll = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=None,
            limit=100,
            with_payload=True
        )
        while scroll[0]:
            for point in scroll[0]:
                if "filename" in point.payload:
                    indexed.add(point.payload["filename"])
            scroll = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                offset=scroll[1],
                limit=100,
                with_payload=True
            )
        return indexed
    except Exception as e:
        print(f"⚠️ Error consultando Qdrant: {e}")
        return set()


def compress_page_ranges(pages):
    """Convierte lista de páginas en rango comprimido tipo '1-3,5,7-9'."""
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


# ===============================
# Carga e indexación de PDFs
# ===============================
def load_pdfs_azure():
    """
    Descarga de Azure solo los PDFs que no están en Qdrant.
    Indexa en Qdrant los nuevos PDFs encontrados.
    """
    pdfs = []
    metadatas = []

    indexed_filenames = get_indexed_filenames()
    print(f"📂 Archivos ya indexados en Qdrant: {len(indexed_filenames)}")

    try:
        blobs = list(container_client.list_blobs(name_starts_with="BD_Knowledge"))
    except Exception as e:
        raise RuntimeError(f"❌ Error al listar blobs en BD_Knowledge: {e}")

    new_pdfs_to_index = []

    for blob in blobs:
        if not blob.name.endswith(".pdf"):
            continue

        filename = os.path.basename(blob.name)
        if filename in indexed_filenames:
            print(f"✅ Ya existe en Qdrant, omitiendo descarga: {filename}")
            continue

        print(f"📥 Descargando nuevo PDF: {filename}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                downloader = container_client.download_blob(blob.name)
                downloader.download_to_stream(temp_pdf)
                temp_pdf_path = temp_pdf.name
        except Exception as e:
            print(f"⚠️ Error al descargar {filename}: {e}")
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
                pdfs.append(pdf_data)
                metadatas.append(metadata)
                new_pdfs_to_index.append(pdf_data)

        except Exception as e:
            print(f"⚠️ Error procesando PDF {filename}: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

    # Indexar en Qdrant solo los nuevos
    if new_pdfs_to_index:
        index_pdf_chunks(new_pdfs_to_index)
        print(f"📌 Indexados {len(new_pdfs_to_index)} nuevos PDFs en Qdrant")

    return pdfs, metadatas

