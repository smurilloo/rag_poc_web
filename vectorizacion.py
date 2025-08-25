import os
import time
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1", "").strip().strip('"')
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')

if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("❌ Faltan variables de entorno QDRANT_API_KEY1 o QDRANT_URL")

try:
    QDRANT_URL = urllib.parse.unquote(QDRANT_URL).replace('"', '').strip()
    parsed_url = urllib.parse.urlparse(QDRANT_URL)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("❌ QDRANT_URL no tiene estructura válida")
    logger.info(f"URL validada correctamente: {QDRANT_URL}")
except Exception as e:
    logger.error(f"❌ Error al validar QDRANT_URL: {e}")
    raise

COLLECTION_NAME = "vector_bd"
encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
client.get_collections()  # validación explícita

def ensure_collection():
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

def wait_until_collection_ready(max_retries=10, delay=2):
    for _ in range(max_retries):
        try:
            info = client.get_collection(collection_name=COLLECTION_NAME)
            if info.status == "green":
                return
        except Exception:
            pass
        time.sleep(delay)
    raise TimeoutError(f"❌ Colección '{COLLECTION_NAME}' no lista después de {max_retries*delay}s")

def get_id(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**16)

def _upsert_points(points, batch_size=50):
    if not points:
        logger.warning("⚠️ No hay puntos para insertar.")
        return
    try:
        for i in range(0, len(points), batch_size):
            client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
    except Exception as e:
        logger.error(f"❌ Error al insertar puntos: {e}")

def _filter_existing_ids(ids):
    """Verifica cuáles IDs ya existen en la colección y devuelve solo los nuevos"""
    existing_ids = set()
    try:
        # Chequeo rápido por IDs en batches
        for i in range(0, len(ids), 100):
            resp = client.scroll(collection_name=COLLECTION_NAME, filter=None, ids=ids[i:i+100])
            existing_ids.update(p.id for p in resp.points)
    except Exception as e:
        logger.error(f"⚠️ No se pudo filtrar IDs existentes: {e}")
    return [id_ for id_ in ids if id_ not in existing_ids]

def index_pdf_chunks(pdf_data):
    ensure_collection()
    points = []
    ids_to_check = []
    for doc in pdf_data:
        for page in doc["pages_texts"]:
            content = page["text"].strip()
            if not content:
                continue
            uid = get_id(content)
            ids_to_check.append(uid)
            points.append(PointStruct(
                id=uid,
                vector=encoder.encode(content).tolist(),
                payload={
                    "type": "pdf",
                    "filename": doc["filename"],
                    "title": doc["title"],
                    "page": page["page"],
                    "content": content
                }
            ))
    # Solo insertar puntos nuevos
    new_points = [p for p in points if p.id in _filter_existing_ids(ids_to_check)]
    _upsert_points(new_points)

def index_web_papers(web_papers):
    ensure_collection()
    points = []
    ids_to_check = []
    for paper in web_papers:
        snippet = paper["snippet"].strip()
        if not snippet:
            continue
        uid = get_id(paper["url"])
        ids_to_check.append(uid)
        points.append(PointStruct(
            id=uid,
            vector=encoder.encode(snippet).tolist(),
            payload={
                "type": "web",
                "url": paper["url"],
                "title": paper["title"],
                "page": 1,
                "score": paper.get("score", 0),
                "content": snippet
            }
        ))
    new_points = [p for p in points if p.id in _filter_existing_ids(ids_to_check)]
    _upsert_points(new_points)

def search_qdrant(query, top_k=5):
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

__all__ = [
    "client",
    "COLLECTION_NAME",
    "index_pdf_chunks",
    "index_web_papers",
    "ensure_collection",
    "search_qdrant"
]
