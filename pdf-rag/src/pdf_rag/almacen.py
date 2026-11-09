"""Base vectorial ChromaDB persistente en disco."""
import chromadb
from chromadb.utils import embedding_functions

from .config import MODELO_EMBEDDINGS, RUTA_CHROMA


def crear_cliente(ruta: str = RUTA_CHROMA) -> chromadb.ClientAPI:
    """Crea (o abre) una base vectorial persistente en disco."""
    return chromadb.PersistentClient(path=ruta)


def funcion_embeddings():
    """Modelo multilingüe: funciona bien con textos en español."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODELO_EMBEDDINGS)


def obtener_coleccion(cliente, nombre: str, funcion_emb=None):
    """Devuelve una colección que mide la similitud con la distancia coseno."""
    return cliente.get_or_create_collection(
        name=nombre,
        embedding_function=funcion_emb or funcion_embeddings(),
        metadata={"hnsw:space": "cosine"},
    )
