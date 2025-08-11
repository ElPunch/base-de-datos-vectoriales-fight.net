import os
import shutil
import random
from datetime import datetime, timedelta
from pprint import pprint
import numpy as np
from tqdm import tqdm

# chromadb: intentamos soportar tanto PersistentClient como Client+Settings por compatibilidad
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    raise RuntimeError("chromadb no está instalado o no se pudo importar. Instala las dependencias.") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("sentence-transformers no está instalado o no se pudo importar. Instala las dependencias.") from e

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


# ---------------------------
# CONFIG
# ---------------------------
CHROMA_DIR = "chroma_collections"   # carpeta persistente que se generará (entréga)
ZIP_NAME = "chroma_collections_zip"
COLLECTION_NAME = "eventos"
N_DOCS = 80   # >= 75
BATCH_SIZE = 32
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sustitúyelo si deseas otro modelo


# ---------------------------
# Funciones utilitarias
# ---------------------------
def create_chromadb_client(path: str):
    """
    Crea un cliente persistente de chromadb.
    Intenta PersistentClient, si no existe usa Client(Settings(...)).
    """
    # Primera opción: PersistentClient (API antigua/nueva según versión)
    try:
        client = chromadb.PersistentClient(path=path)
        print("Usando chromadb.PersistentClient(path=...)")
        return client
    except Exception:
        pass

    # Segunda opción: chromadb.Client con Settings
    try:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=path)
        client = chromadb.Client(settings=settings)
        print("Usando chromadb.Client(Settings(...))")
        return client
    except Exception as e:
        raise RuntimeError("No se pudo crear el cliente chromadb. Revisa la versión de chromadb.") from e


def gen_event(i, disciplinas, ciudades, organizadores, usuarios):
    nombre = f"Combate #{i} - {random.choice(disciplinas)}"
    descripcion = (
        f"Evento deportivo de {random.choice(disciplinas)} con peleadores nacionales e internacionales. "
        "Cartel completo con combates estelares, semiestelares y peleas amateurs. "
        "Entrada general disponible en taquilla y en línea."
    )
    localizacion = random.choice(ciudades)
    organizador = random.choice(organizadores)
    disciplina = random.choice(disciplinas)
    usuario = random.choice(usuarios)
    fecha = (datetime.now() + timedelta(days=random.randint(-365, 365))).isoformat()
    metadata = {
        "nombre": nombre,
        "localizacion": localizacion,
        "organizador": organizador,
        "disciplina": disciplina,
        "usuario": usuario,
        "fecha": fecha,
        "source": "synthetic_eventos"
    }
    document = (
        f"{nombre}\n{descripcion}\nUbicación: {localizacion}\nOrganizador: {organizador}\n"
        f"Disciplina: {disciplina}\nUsuario: {usuario}\nFecha: {fecha}"
    )
    return document, metadata


def compute_embeddings(model, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        # Aseguramos tipo lista plano (no numpy) para chroma
        if hasattr(emb, "tolist"):
            emb_list = emb.tolist()
        else:
            emb_list = [list(e) for e in emb]
        embeddings.extend(emb_list)
    return embeddings


def top_k_euclidean(query_text, model, stored_embs, stored_ids, stored_docs, stored_metadatas, k=5):
    q_emb = model.encode([query_text])[0].reshape(1, -1)
    dists = pairwise_distances(q_emb, stored_embs, metric='euclidean')[0]
    idxs = np.argsort(dists)[:k]
    results = []
    for idx in idxs:
        results.append({
            "id": stored_ids[idx],
            "doc": stored_docs[idx],
            "metadata": stored_metadatas[idx],
            "distance": float(dists[idx])
        })
    return results


def top_k_cosine(query_text, model, stored_embs, stored_ids, stored_docs, stored_metadatas, k=5):
    q_emb = model.encode([query_text])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, stored_embs)[0]
    idxs = np.argsort(-sims)[:k]
    results = []
    for idx in idxs:
        results.append({
            "id": stored_ids[idx],
            "doc": stored_docs[idx],
            "metadata": stored_metadatas[idx],
            "score": float(sims[idx])
        })
    return results


# ---------------------------
# MAIN
# ---------------------------
def main():
    random.seed(42)

    # listas para generar eventos
    disciplinas = ["Boxeo", "MMA", "Kickboxing", "Lucha Libre", "Jiu-Jitsu", "Taekwondo"]
    ciudades = ["Ciudad de México", "Guadalajara", "Monterrey", "Puebla", "Toluca", "Tijuana", "León"]
    organizadores = ["Promotora Alpha", "FightNet Events", "Liga Nacional", "Promotor Omega", "Asociación Deportiva"]
    usuarios = [f"user_{i}" for i in range(1, 21)]

    # 1) Generar documentos
    documents = []
    metadatas = []
    ids = []

    for i in range(1, N_DOCS + 1):
        doc, meta = gen_event(i, disciplinas, ciudades, organizadores, usuarios)
        documents.append(doc)
        metadatas.append(meta)
        ids.append(str(i))

    print(f"Generados {len(documents)} documentos (>=75).")

    # 2) Cargar modelo de embeddings
    print("Cargando modelo de embeddings:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 3) Crear cliente chromadb persistente
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR, exist_ok=True)
    client = create_chromadb_client(CHROMA_DIR)

    # 4) Crear o obtener colección
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        # Algunas versiones devuelven diferente
        collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Colección lista: {COLLECTION_NAME}")

    # 5) Calcular embeddings y añadir a chroma
    print("Calculando embeddings (batches)...")
    embeddings = compute_embeddings(model, documents, batch_size=BATCH_SIZE)
    print("Añadiendo documentos y embeddings a la colección (persistente)...")
    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    # intentar persistir si la API lo permite
    try:
        if hasattr(client, "persist"):
            client.persist()
    except Exception:
        pass

    # 6) Recuperar todo para búsquedas manuales
    all_data = collection.get(include=["ids", "documents", "metadatas", "embeddings"])
    stored_ids = all_data["ids"]
    stored_docs = all_data["documents"]
    stored_metadatas = all_data["metadatas"]
    stored_embs = np.array(all_data["embeddings"])
    print("Datos recuperados: ids", len(stored_ids), "embeddings shape", stored_embs.shape)

    # 7) Ejemplos Euclidiano (5)
    queries_eu = [
        "cartel con pelea estelar de MMA entre peso welter",
        "evento de boxeo en Ciudad de México con entrada en línea",
        "jornada amateur de jiu-jitsu para cinturones blancos y azules",
        "promotora organizando campeonato nacional de taekwondo",
        "lucha libre con luchadores internacionales y espectáculo"
    ]
    print("\n=== Resultados Euclidiano (top-5) ===")
    for q in queries_eu:
        res = top_k_euclidean(q, model, stored_embs, stored_ids, stored_docs, stored_metadatas, k=5)
        print(f"\nConsulta: {q}")
        for r in res:
            print(f" id={r['id']} | distancia={r['distance']:.4f} | loc={r['metadata']['localizacion']} | org={r['metadata']['organizador']}")

    # 8) Ejemplos Coseno (5)
    queries_cos = [
        "cartel MMA internacional con peleas principales",
        "evento boxeo en Guadalajara con venta por internet",
        "torneo jiu-jitsu amateur y profesional",
        "competencia nacional taekwondo sede Toluca",
        "espectáculo lucha libre familiar"
    ]
    print("\n=== Resultados Coseno (top-5) ===")
    for q in queries_cos:
        res = top_k_cosine(q, model, stored_embs, stored_ids, stored_docs, stored_metadatas, k=5)
        print(f"\nConsulta: {q}")
        for r in res:
            print(f" id={r['id']} | score={r['score']:.4f} | disciplina={r['metadata']['disciplina']} | fecha={r['metadata']['fecha']}")

    # 9) Empaquetar carpeta chroma para entrega
    print("\nComprimiendo carpeta de Chroma en ZIP...")
    # eliminamos zip existente si existe
    zipfile = ZIP_NAME + ".zip"
    if os.path.exists(zipfile):
        try:
            os.remove(zipfile)
        except Exception:
            pass
    shutil.make_archive(ZIP_NAME, 'zip', CHROMA_DIR)
    print("Generado archivo:", zipfile)
    print("\n¡Listo! Entrega:", zipfile)
    print(f"La carpeta persistente de Chroma está en: {os.path.abspath(CHROMA_DIR)}")


if __name__ == "__main__":
    main()