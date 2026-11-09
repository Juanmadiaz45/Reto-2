"""Pipeline RAG: recuperar fragmentos del PDF en ChromaDB y responder con el LLM."""
from langsmith import traceable

from .almacen import crear_cliente, obtener_coleccion
from .cargador import indexar_pdf
from .config import COLECCION, RUTA_CHROMA
from .llm import llamar_llm

SISTEMA = (
    "Usted es un asistente que responde preguntas sobre un documento PDF. "
    "Responda en español, de forma concisa, usando SOLO la información de los fragmentos "
    "proporcionados. Si la respuesta no está en los fragmentos, diga que el documento "
    "no contiene esa información. Cite las páginas entre corchetes, por ejemplo [p. 3]."
)


class RagPDF:
    def __init__(self, ruta: str = RUTA_CHROMA, coleccion: str = COLECCION, funcion_emb=None):
        self.coleccion = obtener_coleccion(crear_cliente(ruta), coleccion, funcion_emb)

    def indexar(self, ruta_pdf: str) -> int:
        return indexar_pdf(self.coleccion, ruta_pdf)

    @traceable(run_type="retriever", name="recuperar")
    def recuperar(self, pregunta: str, k: int = 4) -> list[dict]:
        """Devuelve los k fragmentos más parecidos a la pregunta."""
        total = self.coleccion.count()
        if total == 0:
            return []
        r = self.coleccion.query(query_texts=[pregunta], n_results=min(k, total))
        # Formato de documentos que LangSmith muestra en las trazas de tipo retriever
        return [
            {"page_content": doc, "metadata": {**meta, "distancia": dist}, "type": "Document"}
            for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

    @traceable(run_type="chain", name="rag_pdf")
    def responder(self, pregunta: str, k: int = 4) -> dict:
        """Responde la pregunta y devuelve también los contextos usados (necesarios para RAGAS)."""
        fragmentos = self.recuperar(pregunta, k)
        if not fragmentos:
            return {"respuesta": "No hay documentos indexados. Use primero: pdf-rag indexar <pdf>",
                    "contextos": []}
        contexto = "\n\n".join(
            f"[p. {f['metadata']['pagina']}] {f['page_content']}" for f in fragmentos
        )
        mensaje = f"Fragmentos del documento:\n\n{contexto}\n\nPregunta: {pregunta}"
        respuesta = llamar_llm(SISTEMA, [{"role": "user", "content": mensaje}])
        return {"respuesta": respuesta, "contextos": [f["page_content"] for f in fragmentos]}
