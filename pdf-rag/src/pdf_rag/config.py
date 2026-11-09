"""Configuración común: rutas, modelos, precios y variables de entorno."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

RUTA_CHROMA = str(BASE_DIR / "datos" / "chroma")
RUTA_RESULTADOS = BASE_DIR / "resultados"
COLECCION = os.environ.get("PDF_RAG_COLECCION", "documentos")

MODELO = os.environ.get("PDF_RAG_MODELO", "openai/gpt-oss-120b")              # genera las respuestas
MODELO_EVALUADOR = os.environ.get("PDF_RAG_EVALUADOR", "openai/gpt-oss-120b")  # juez de RAGAS
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

# Precios en USD por millón de tokens (entrada, salida). Fuente: console.groq.com/docs/models
PRECIOS = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}

PROYECTO_LANGSMITH = os.environ.setdefault("LANGSMITH_PROJECT", "pdf-rag")


def calcular_costo(modelo: str, tokens_entrada: int, tokens_salida: int) -> tuple[float, float]:
    """Devuelve (costo_entrada, costo_salida) en USD."""
    p_entrada, p_salida = PRECIOS.get(modelo, (0.0, 0.0))
    return tokens_entrada * p_entrada / 1_000_000, tokens_salida * p_salida / 1_000_000


def langsmith_activo() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY")) and \
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
