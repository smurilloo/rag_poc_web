import os
import tempfile
from azure.storage.blob import ContainerClient
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models
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
# Setup de índices en Qdrant
# ===============================
def ensure_indexes():
    """
    Garantiza que 'filename' tenga índice de tipo keyword en Qdrant.
    """
    try:
        collection_info = qdrant_client.get_collection(QDRANT_COLLECTION)
        payload_schema = getattr(collection_info, "payload_schema", {}) or {}

        if "filename" not in payload_schema:
            qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="filename",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            print("📌 Índice 'filename' creado en Qdrant.")
        else:
            print("✅ Índice 'filename' ya existe en Qdrant.")
    except Exception as e:
        print(f"⚠️ Error verificando/creando índice 'filename': {e}")

# Crear índice al inicio
ensure_indexes()

# ===============================
# Utilidades
# ===============================
def is_filename_indexed(filename: str) -> bool:
    """
    Verifica en Qdrant si un filename ya está indexado.
    Retorna True si existe, False si no.
    """
    try:
        result, _ = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename)
                )]
            ),
            limit=1,
            with_payload=False
        )
        return len(result) > 0
    except Exception as e:
        print(f"⚠️ Error verificando filename en Qdrant: {e}")
        return False


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
    new_pdfs_to_index = []

    try:
        blobs = list(container_client.list_blobs(name_starts_with="BD_Knowledge"))
    except Exception as e:
        raise RuntimeError(f"❌ Error al listar blobs en BD_Knowledge: {e}")

    for blob in blobs:
        if not blob.name.endswith(".pdf"):
            continue

        filename = os.path.basename(blob.name)

        # ⚡ Consulta puntual a Qdrant
        if is_filename_indexed(filename):
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
