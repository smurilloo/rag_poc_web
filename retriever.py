# Este código lee todos los archivos PDF en la carpeta del proyecto, extrae el texto de cada página,
# identifica los títulos y organiza la información para que pueda ser usada en respuestas o análisis.
# También resume las páginas leídas en rangos para mostrar de forma clara qué páginas se usaron.

import os
import tempfile
from io import BytesIO
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader

# ------------------------
# CONFIGURACIÓN DE KEY VAULT
# ------------------------

# Nombre del Key Vault
KEY_VAULT_NAME = "mi-keyvault"
SECRET_NAME = "AzureBlobSasToken"

# Construir la URL del Key Vault
KV_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net"

# Autenticación usando el contexto 
credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url=KV_URI, credential=credential)

# Obtener la SAS Token del secreto
sas_token = secret_client.get_secret(SECRET_NAME).value

# ------------------------
# CONFIGURACIÓN DE BLOB STORAGE
# ------------------------

# Usar la SAS Token obtenida para construir la cadena de conexión
AZURE_SAS_URL = (
    "BlobEndpoint=https://testingmlai.blob.core.windows.net/;"
    "QueueEndpoint=https://testingmlai.queue.core.windows.net/;"
    "FileEndpoint=https://testingmlai.file.core.windows.net/;"
    "TableEndpoint=https://testingmlai.table.core.windows.net/;"
    f"SharedAccessSignature={sas_token}"
)

CONTAINER_NAME = "azureml-blobstore-b70b1075-9c59-4b11-8dc1-7f0c1649da09"
BLOB_DIR = "RAG_Source_Storage"

# Inicializar clientes de Azure Blob
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
        downloader = container_client.download_blob(blob.name)
        pdf_stream = BytesIO(downloader.readall())

        reader = PdfReader(pdf_stream)
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
    # Implementar carga desde Google Cloud
    return [], []

def load_pdfs_aws():
    # Implementar carga desde AWS
    return [], []
