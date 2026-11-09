# pdf-rag

RAG sobre un PDF con ChromaDB y Groq. La precisión se mide con RAGAS y el costo se registra en LangSmith. La explicación completa y los resultados están en el [README principal](../README.md).

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Rutas, modelos, tabla de precios y carga de `.env` |
| `almacen.py` | Cliente persistente de ChromaDB y función de embeddings |
| `cargador.py` | Lee el PDF, lo divide en fragmentos y los indexa |
| `llm.py` | Llamada a Groq registrada en LangSmith con tokens y costo |
| `rag.py` | Recupera fragmentos y responde citando páginas |
| `evaluacion.py` | Genera el dataset, evalúa con RAGAS, envía el feedback y consulta el costo en LangSmith |
| `main.py` | Comandos de consola |

## Uso

```bash
uv sync
cp .env.example .env      # GROQ_API_KEY y LANGSMITH_API_KEY

uv run pdf-rag indexar datos/tardigrados.pdf
uv run pdf-rag preguntar "¿De qué trata el documento?"
uv run pdf-rag chat
uv run pdf-rag generar-dataset -n 10
uv run pdf-rag evaluar
uv run pdf-rag costo
```

`datos/dataset.json` tiene la forma `[{"pregunta": "...", "referencia": "..."}]`. Conviene revisar las preguntas generadas antes de evaluar, porque la referencia es contra lo que se mide la respuesta.

Cada evaluación guarda en `resultados/` un CSV con las notas por pregunta y un JSON con los promedios y los costos. El modelo y el juez se pueden cambiar con `PDF_RAG_MODELO` y `PDF_RAG_EVALUADOR`; si se usa otro modelo hay que agregar su precio en `PRECIOS` dentro de `config.py`.
