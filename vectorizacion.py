import os
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------
# Configuración desde variables de entorno
# -----------------------------
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY1")
QDRANT_URL = os.getenv("QDRANT_URL")  # Opcional si usas una URL fija
COLLECTION_NAME = "vector_bd"

if not QDRANT_API_KEY:
    raise ValueError("❌ Falta QDRANT_API_KEY1")

encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(
    url=QDRANT_URL or "https://1730be49-201a-4ad4-ac38-0c5e2e86d0b9.us-west-2-0.aws.cloud.qdrant.io:6333",
    api_key=QDRANT_API_KEY,
    prefer_grpc=False,
    timeout=120
)

def ensure_collection():
    if not client.collection_exists(COLLECTION_NAME):
        print(f"📁 Colección '{COLLECTION_NAME}' no existe. Creándola...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            )
        )
    else:
        print(f"📁 Colección '{COLLECTION_NAME}' ya existe.")

# Asegurar colección al iniciar
ensure_collection()

def get_id(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (10**16)

def index_pdf_chunks(pdf_data):
    ensure_collection()
    points = []
    for doc in pdf_data:
        for page in doc["pages_texts"]:
            content = page["text"]
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
    ensure_collection()
    points = []
    for paper in web_papers:
        snippet = paper["snippet"]
        for i in range(0, len(snippet), 500):
            chunk = snippet[i:i+500]
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
    if not points:
        return
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def get_first_k_points(k=10):
    ensure_collection()
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=k)
    return [
        {
            "id": point.id,
            "payload": point.payload
        }
        for point in points
    ]

def cleanup_collection(limit=20):
    if not client.collection_exists(COLLECTION_NAME):
        print("⚠️ No se puede limpiar: colección no existe.")
        return

    current_count = client.count(collection_name=COLLECTION_NAME).count
    if current_count > limit:
        print(f"🧹 Limpiando colección. Registros actuales: {current_count}")
        scroll_offset = None
        deleted = 0
        while deleted + limit < current_count:
            points, scroll_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=scroll_offset
            )
            ids_to_delete = [p.id for p in points]
            if not ids_to_delete:
                break
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=ids_to_delete)
            )
            deleted += len(ids_to_delete)
        print(f"✅ Limpieza completada. Registros eliminados: {deleted}")
    else:
        print("📦 No hay suficientes registros para limpiar.")

def delete_collection():
    try:
        if client.collection_exists(COLLECTION_NAME):
            print(f"🧨 Borrando colección completa: {COLLECTION_NAME}")
            client.delete_collection(collection_name=COLLECTION_NAME)
            print("✅ Colección eliminada.")
        else:
            print(f"ℹ️ Colección '{COLLECTION_NAME}' ya no existe. Nada que borrar.")
    except Exception as e:
        print(f"❌ Error al borrar la colección: {e}")

__all__ = [
    "client",
    "COLLECTION_NAME",
    "index_pdf_chunks",
    "index_web_papers",
    "get_first_k_points",
    "cleanup_collection",
    "delete_collection"
]
