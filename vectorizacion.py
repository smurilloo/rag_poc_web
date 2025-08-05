import os
import time
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
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

try:
    QDRANT_URL = urllib.parse.unquote(QDRANT_URL).replace('"', '').strip()
    parsed = urllib.parse.urlparse(QDRANT_URL)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("❌ URL inválida")
    logger.info(f"✅ URL validada: {QDRANT_URL}")
except Exception as e:
    raise ValueError(f"❌ Error al validar URL: {e}")

COLLECTION_NAME = "vector_bd"

# === Inicialización ===
encoder = SentenceTransformer("all-MiniLM-L6-v2")

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.get_collections()
    print(f"✅ Conectado a Qdrant en {QDRANT_URL}")
except Exception as e:
    raise ConnectionError(f"❌ Error al conectar con Qdrant: {e}")


def ensure_collection():
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"📁 Creando colección '{COLLECTION_NAME}'...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
        wait_until_collection_ready()
        print("✅ Colección creada.")
    else:
        print(f"📁 Colección '{COLLECTION_NAME}' ya existe.")


def wait_until_collection_ready(max_retries=10, delay=2):
    for _ in range(max_retries):
        try:
            if client.get_collection(collection_name=COLLECTION_NAME).status == "green":
                return
        except Exception:
            pass
        time.sleep(delay)
    raise TimeoutError("❌ La colección no estuvo lista a tiempo.")


def _hash_id(text: str) -> str:
    """
    Genera un hash SHA-256 único basado en el contenido normalizado.
    """
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _content_exists(content_hash: str) -> bool:
    """
    Verifica si el contenido ya existe en la BD vectorial usando su hash.
    """
    try:
        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[content_hash],
            with_payload=False,
            with_vectors=False
        )
        return len(result) > 0
    except Exception as e:
        logger.error(f"❌ Error al verificar existencia: {e}")
        return False


def index_pdf_chunks(pdf_data):
    ensure_collection()
    points = []
    for doc in pdf_data:
        for page in doc["pages_texts"]:
            content = page["text"]
            if not content.strip():
                continue
            content_hash = _hash_id(content)
            if _content_exists(content_hash):
                continue
            vector = encoder.encode(content).tolist()
            metadata = {
                "type": "pdf",
                "filename": doc["filename"],
                "title": doc["title"],
                "page": page["page"],
                "content": content
            }
            points.append(PointStruct(id=content_hash, vector=vector, payload=metadata))
    _upsert_points(points)


def index_web_papers(web_papers):
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
            content_hash = _hash_id(chunk)
            if _content_exists(content_hash):
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
            points.append(PointStruct(id=content_hash, vector=vector, payload=metadata))
    _upsert_points(points)


def _upsert_points(points):
    if not points:
        print("⚠️ No hay puntos nuevos para insertar.")
        return
    try:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ {len(points)} puntos nuevos insertados.")
    except Exception as e:
        logger.error(f"❌ Error al insertar: {e}")


__all__ = [
    "client",
    "COLLECTION_NAME",
    "index_pdf_chunks",
    "index_web_papers",
    "ensure_collection"
]
