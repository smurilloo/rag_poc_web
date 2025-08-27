import os
import time
import hashlib
import logging
import urllib.parse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ===============================
# Configuración logging
# ===============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Variables de entorno
# ===============================
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1", "").strip().strip('"')
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')

if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("❌ Faltan variables de entorno QDRANT_API_KEY1 o QDRANT_URL")

try:
    QDRANT_URL = urllib.parse.unquote(QDRANT_URL).replace('"', '').strip()
    parsed_url = urllib.parse.urlparse(QDRANT_URL)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("❌ QDRANT_URL no tiene estructura válida")
    logger.info(f"✅ URL validada correctamente: {QDRANT_URL}")
except Exception as e:
    logger.error(f"❌ Error al validar QDRANT_URL: {e}")
    raise

# ===============================
# Configuración Qdrant y Encoder
# ===============================
COLLECTION_NAME = "vector_bd"
encoder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Validación inicial
try:
    client.get_collections()
    logger.info("✅ Cliente Qdrant inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando cliente Qdrant: {e}")
    raise

# ===============================
# Funciones de soporte
# ===============================
def ensure_collection():
    """Crea la colección si no existe"""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        logger.info(f"📁 Colección '{COLLECTION_NAME}' no existe. Creando...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
        wait_until_collection_ready()
        logger.info(f"✅ Colección '{COLLECTION_NAME}' creada y lista")

def wait_until_collection_ready(max_retries=10, delay=2):
    """Espera hasta que la colección esté lista"""
    for _ in range(max_retries):
        try:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            if info.status == "green":
                return
        except Exception:
            pass
        time.sleep(delay)
    raise TimeoutError(f"❌ Colección '{COLLECTION_NAME}' no lista después de {max_retries*delay}s")

def get_id(text: str) -> int:
    """Genera un ID único a partir del contenido (PDF page text o URL)"""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**16)

def _upsert_points(points: list[PointStruct], batch_size=50):
    """Inserta los puntos en batches"""
    if not points:
        logger.info("ℹ️ No hay puntos nuevos para insertar")
        return
    try:
        for i in range(0, len(points), batch_size):
            client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
        logger.info(f"✅ Se insertaron {len(points)} puntos en Qdrant")
    except Exception as e:
        logger.error(f"❌ Error al insertar puntos: {e}")

def _filter_existing_ids(ids: list[int]) -> set[int]:
    """Devuelve los IDs que ya existen en la colección"""
    existing_ids = set()
    try:
        for i in range(0, len(ids), 100):
            batch_ids = ids[i:i+100]
            resp = client.retrieve(collection_name=COLLECTION_NAME, ids=batch_ids)
            existing_ids.update(p.id for p in resp)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo filtrar IDs existentes: {e}")
    return existing_ids

# ===============================
# Funciones principales
# ===============================
def index_pdf_chunks(pdf_data: list[dict]):
    """Indexa el contenido de PDFs en Qdrant"""
    ensure_collection()

    # Calcular IDs de todo el batch
    id_to_content = {}
    for doc in pdf_data:
        for page in doc["pages_texts"]:
            content = page["text"].strip()
            if not content:
                continue
            uid = get_id(content)
            id_to_content[uid] = {
                "filename": doc["filename"],
                "title": doc["title"],
                "page": page["page"],
                "content": content,
            }

    # Filtrar solo los que NO existen
    existing_ids = _filter_existing_ids(list(id_to_content.keys()))
    new_ids = set(id_to_content.keys()) - existing_ids

    # Generar solo los puntos nuevos
    new_points = [
        PointStruct(
            id=uid,
            vector=encoder.encode(data["content"]).tolist(),
            payload={
                "type": "pdf",
                "filename": data["filename"],
                "title": data["title"],
                "page": data["page"],
                "content": data["content"]
            }
        )
        for uid, data in id_to_content.items() if uid in new_ids
    ]

    _upsert_points(new_points)

def index_web_papers(web_papers: list[dict]):
    """Indexa papers web en Qdrant"""
    ensure_collection()

    id_to_paper = {}
    for paper in web_papers:
        snippet = paper["snippet"].strip()
        if not snippet:
            continue
        uid = get_id(paper["url"])
        id_to_paper[uid] = {
            "url": paper["url"],
            "title": paper["title"],
            "page": 1,
            "score": paper.get("score", 0),
            "content": snippet,
        }

    existing_ids = _filter_existing_ids(list(id_to_paper.keys()))
    new_ids = set(id_to_paper.keys()) - existing_ids

    new_points = [
        PointStruct(
            id=uid,
            vector=encoder.encode(data["content"]).tolist(),
            payload={
                "type": "web",
                "url": data["url"],
                "title": data["title"],
                "page": data["page"],
                "score": data["score"],
                "content": data["content"]
            }
        )
        for uid, data in id_to_paper.items() if uid in new_ids
    ]

    _upsert_points(new_points)

def search_qdrant(query: str, top_k: int = 5) -> list[dict]:
    """Busca en Qdrant y devuelve resultados"""
    query_vector = encoder.encode(query).tolist()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k)

    results = []
    for hit in hits:
        payload = hit.payload
        results.append({
            "type": payload["type"],
            "source": payload.get("filename", payload.get("url")),
            "title": payload["title"],
            "page": payload["page"],
            "score": hit.score,
            "content": payload.get("content", "")
        })
    return results

# ===============================
# Exports
# ===============================
__all__ = [
    "client",
    "COLLECTION_NAME",
    "index_pdf_chunks",
    "index_web_papers",
    "ensure_collection",
    "search_qdrant"
]
