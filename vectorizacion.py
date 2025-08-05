import os
import time
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from sentence_transformers import SentenceTransformer
import urllib.parse
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Validación segura de variables de entorno ===
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1", "").strip().strip('"')
if not QDRANT_API_KEY:
    raise ValueError("❌ Falta la variable de entorno QDRANT_API_KEY1")

QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')
if not QDRANT_URL:
    raise ValueError("❌ Falta la variable de entorno QDRANT_URL")

# Limpieza adicional de la URL para evitar caracteres de escape incorrectos
try:
    # Decodificar URL para eliminar cualquier carácter de escape
    QDRANT_URL = urllib.parse.unquote(QDRANT_URL)
    # Eliminar comillas dobles y espacios en blanco adicionales
    QDRANT_URL = QDRANT_URL.replace('"', '').strip()
    # Verificar que la URL no esté vacía después de la limpieza
    if not QDRANT_URL:
        raise ValueError("❌ La variable de entorno QDRANT_URL no es válida después de limpieza")
    
    # Validar la estructura de la URL
    parsed_url = urllib.parse.urlparse(QDRANT_URL)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("❌ La variable de entorno QDRANT_URL no tiene una estructura de URL válida")
    
    logger.info(f"URL validada correctamente: {QDRANT_URL}")
except Exception as e:
    logger.error(f"❌ Error al limpiar o validar la URL: {e}")
    raise ValueError(f"❌ Error al limpiar o validar la URL: {e}")

COLLECTION_NAME = "vector_bd"

# === Inicialización del encoder y cliente Qdrant ===
encoder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    # Validación explícita de conexión
    client.get_collections()
    print(f"✅ Conectado exitosamente a Qdrant en {QDRANT_URL}")
except Exception as e:
    logger.error(f"❌ No se pudo inicializar o conectar con Qdrant: {e}")
    raise ConnectionError(f"❌ No se pudo inicializar o conectar con Qdrant: {e}")


def ensure_collection():
    """
    Verifica si la colección existe en Qdrant. Si no existe, la crea.
    """
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"📁 Colección '{COLLECTION_NAME}' no existe. Creándola...")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=encoder.get_sentence_embedding_dimension(), 
                    distance=Distance.COSINE,
                )
            )
            print("⏳ Esperando a que la colección esté lista...")
            wait_until_collection_ready(max_retries=10, delay=2)
            print("✅ Colección creada y lista.")
        else:
            print(f"📁 Colección '{COLLECTION_NAME}' ya existe.")
    except Exception as e:
        logger.error(f"❌ Error al verificar o crear la colección: {e}")
        raise RuntimeError(f"❌ Error al verificar o crear la colección: {e}")


def wait_until_collection_ready(max_retries=10, delay=1):
    """
    Espera hasta que la colección esté lista para ser usada.
    """
    for attempt in range(max_retries):
        try:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            if info.status == "green":
                return
        except Exception:
            pass
        time.sleep(delay)
    logger.error(f"❌ La colección '{COLLECTION_NAME}' no estuvo lista después de {max_retries * delay} segundos.")
    raise TimeoutError(f"❌ La colección '{COLLECTION_NAME}' no estuvo lista después de {max_retries * delay} segundos.")


def get_id(text):
    """
    Genera un ID único a partir del texto proporcionado.
    """
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**16)


def index_pdf_chunks(pdf_data):
    """
    Índice de fragmentos de PDF en Qdrant.
    """
    ensure_collection()
    points = []
    for doc in pdf_data:
        for page in doc["pages_texts"]:
            content = page["text"]
            if not content.strip():
                continue
            vector = encoder.encode(content).tolist()
            metadata = {
                "type": "pdf",
                "filename": doc["filename"],
                "title": doc["title"],
                "page": page["page"],
                "content": content
            }
            points.append(PointStruct(id=get_id(content), vector=vector, payload=metadata))
    _upsert_points(points)


def index_web_papers(web_papers):
    """
    Índice de artículos web en Qdrant.
    """
    ensure_collection()
    points = []
    for paper in web_papers:
        snippet = paper["snippet"]
        if not snippet.strip():
            continue
        for i in range(0, len(snippet), 500):
            chunk = snippet[i:i+500]
            if not chunk.strip():
                continue
            vector = encoder.encode(chunk).tolist()
            page_number = i // 500 + 1
            metadata = {
                "type": "web",
                "url": paper["url"],
                "title": paper["title"],
                "page": page_number,
                "score": paper.get("score", 0),
                "content": chunk
            }
            uid = get_id(paper["url"] + str(page_number))
            points.append(PointStruct(id=uid, vector=vector, payload=metadata))
    _upsert_points(points)


def _upsert_points(points):
    """
    Inserta o actualiza los puntos en la colección de Qdrant.
    """
    if not points:
        print("⚠️ No hay puntos para insertar.")
        return
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"✅ {len(points)} puntos insertados en '{COLLECTION_NAME}'.")
    except Exception as e:
        logger.error(f"❌ Error al insertar puntos: {e}")
        if "forbidden" in str(e).lower():
            print("🚫 Acceso prohibido: verifica la API key, el nombre de la colección o permisos.")
        print(f"❌ Error al insertar puntos: {e}")





# Exportar funciones públicas
__all__ = [
    "client",
    "COLLECTION_NAME",
    "index_pdf_chunks",
    "index_web_papers",
    "ensure_collection"
]

