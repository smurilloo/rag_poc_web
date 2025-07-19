# Este código lee todos los archivos PDF en la carpeta del proyecto, extrae el texto de cada página,
# identifica los títulos y organiza la información para que pueda ser usada en respuestas o análisis.
# También resume las páginas leídas en rangos para mostrar de forma clara qué páginas se usaron.

import os
import tempfile
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader

# ----------------------------
# Config segura: Key Vault o variable de entorno
# ----------------------------
KEY_VAULT_NAME = "pocragweb"
SECRET_NAME_STORAGE = "POC-RAG-WEB-BLTBKM1-STORAGE1"
KV_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net"

try:
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=KV_URI, credential=credential)
    sas_token = secret_client.get_secret(SECRET_NAME_STORAGE).value
except Exception as e:
    print(f"⚠️ Key Vault falló: {e}")
    sas_token = os.getenv("AZURE_STORAGE_SAS_TOKEN")

if not sas_token:
    raise ValueError("❌ AZURE_STORAGE_SAS_TOKEN no está configurada ni disponible en Key Vault.")

AZURE_SAS_URL = (
    "BlobEndpoint=https://testingmlai.blob.core.windows.net/;"
    "QueueEndpoint=https://testingmlai.queue.core.windows.net/;"
    "FileEndpoint=https://testingmlai.file.core.windows.net/;"
    "TableEndpoint=https://testingmlai.table.core.windows.net/;"
    f"SharedAccessSignature={sas_token}"
)

CONTAINER_NAME = "pocragweb"
BLOB_DIR = "BD_Knowledge"

blob_service_client = BlobServiceClient.from_connection_string(AZURE_SAS_URL)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def load_pdfs_azure():
    pdfs = []
    metadatas = []

    blobs = container_client.list_blobs(name_starts_with=BLOB_DIR)

    for blob in blobs:
        if not blob.name.endswith(".pdf"):
            continue

        print(f"🔹 Descargando: {blob.name}")

        # Guardar temporalmente el PDF para procesarlo de forma segura
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            downloader = container_client.download_blob(blob.name)
            downloader.download_to_stream(temp_pdf)
            temp_pdf_path = temp_pdf.name

        # Procesar el PDF desde el archivo temporal
        reader = PdfReader(temp_pdf_path)
        pages_texts = []
        pages_numbers = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_texts.append({"page": i + 1, "text": text.strip()})
                pages_numbers.append(i + 1)

        # Eliminar el archivo temporal
        os.remove(temp_pdf_path)

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


def load_pdfs_google():
    return [], []


def load_pdfs_aws():
    return [], []

