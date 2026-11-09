# Reto 2

Integrantes:

- Juan Manuel Díaz Moreno - A00394477
- Santiago Valencia García - A00395902

El reto tiene dos partes:

1. **pdf-rag**: un sistema RAG que lee un PDF, guarda sus fragmentos en ChromaDB y responde preguntas con un LLM de Groq. La precisión se mide con RAGAS y el costo se registra en LangSmith.
2. **proyecto-mcp**: el tutorial de memoria persistente y servidores MCP en VS Code (servidores memory, notas y fetch), más un servidor MCP propio que consulta en vivo la documentación de LangChain.

Carpetas:

- `pdf-rag/`: RAG sobre PDF con ChromaDB, RAGAS y LangSmith.
- `proyecto-mcp/`: configuración de los servidores MCP para VS Code (`.vscode/mcp.json`).
- `proyecto-mcp/mi-servidor/`: servidor de notas del tutorial.
- `proyecto-mcp/docs-servidor/`: servidor que consulta la documentación de LangChain.

Requisitos: [uv](https://docs.astral.sh/uv/) y Python 3.12 para todo; Node.js y VS Code con GitHub Copilot Chat para la parte MCP.

## Parte 1: pdf-rag

### Qué hace

1. Lee el PDF con `pypdf` y lo corta en fragmentos de 1000 caracteres con 200 de solape, guardando la página de cada uno.
2. Guarda los fragmentos en una colección persistente de ChromaDB con embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`) y distancia coseno. Los ids dependen del contenido, así que volver a indexar el mismo PDF no duplica datos.
3. Ante una pregunta, recupera los 4 fragmentos más parecidos y se los pasa al modelo `openai/gpt-oss-120b` de Groq, que responde citando las páginas.
4. Para evaluar, genera preguntas con su respuesta de referencia a partir del PDF, las responde con el RAG y le pasa todo a RAGAS.
5. Cada llamada al LLM se envía a LangSmith con sus tokens y su costo en dólares, y las notas de RAGAS quedan como feedback en la traza de cada pregunta. Así la precisión y el costo se ven juntos.

### Cómo usarlo

```bash
cd pdf-rag
uv sync
cp .env.example .env      # completar GROQ_API_KEY y LANGSMITH_API_KEY

uv run pdf-rag indexar datos/tardigrados.pdf
uv run pdf-rag preguntar "¿Por qué los tardígrados son tan resistentes?"
uv run pdf-rag generar-dataset -n 4
uv run pdf-rag evaluar
uv run pdf-rag costo      # costo acumulado del proyecto en LangSmith
```

Sin `GROQ_API_KEY` el RAG responde en modo simulado, pero la evaluación con RAGAS sí necesita la clave.

### Métricas de RAGAS

| Métrica | Qué mide |
|---|---|
| `llm_context_precision_with_reference` | Si los fragmentos recuperados son relevantes y aparecen primero |
| `context_recall` | Si se recuperó toda la información necesaria |
| `faithfulness` | Si la respuesta se apoya en los fragmentos y no inventa |
| `answer_relevancy` | Si la respuesta atiende la pregunta |
| `factual_correctness` | Si la respuesta coincide con la referencia |

El juez de RAGAS también es `openai/gpt-oss-120b` en Groq.

### Resultados con `tardigrados.pdf`

Evaluación con 4 preguntas generadas del propio PDF (`pdf-rag/resultados/`):

| Métrica | Valor |
|---|---|
| Precisión del contexto | 0.875 |
| Recall del contexto | 1.000 |
| Fidelidad | 1.000 |
| Relevancia de la respuesta | 0.843 |
| Corrección frente a la referencia | 0.600 |

| Costo | Valor |
|---|---|
| RAG, total de 4 preguntas (LangSmith) | USD 0.000769 |
| RAG, por pregunta | USD 0.000192 |
| Evaluación con RAGAS (juez) | USD 0.020053 |

El costo que reporta LangSmith coincide con el que se calcula en local a partir de los tokens. Evaluar sale unas 25 veces más caro que responder, porque cada métrica hace varias llamadas al LLM. Con solo 4 preguntas las métricas todavía no son estables, así que para un PDF más largo conviene generar 15 preguntas o más.

### Problemas que resolvimos

- ragas 0.4.3 falla al importarse con `langchain-community` 0.4, así que fijamos `langchain-community<0.4`.
- Groq limita las peticiones por minuto y algunas evaluaciones daban timeout. Ahora RAGAS hace como máximo 2 peticiones en paralelo y reintenta.
- El endpoint de estadísticas de LangSmith devolvía 429. Ahora el costo se lee de las trazas de cada pregunta, con reintentos, y si LangSmith falla los resultados de RAGAS se guardan igual.
- El generador de preguntas a veces copiaba el fragmento entero como respuesta de referencia. Se ajustó el prompt para pedir 1 o 2 oraciones.

## Parte 2: proyecto-mcp

### El tutorial

Seguimos el tutorial de memoria persistente y servidores MCP en VS Code. En `.vscode/mcp.json` quedan configurados estos servidores, que GitHub Copilot Chat usa en modo Agent:

| Servidor | Qué hace | Cómo se ejecuta |
|---|---|---|
| `memory` | Guarda datos del usuario como un grafo de entidades, relaciones y observaciones en `memoria/memory.jsonl` | `npx @modelcontextprotocol/server-memory` |
| `notas` | Servidor propio en Python con `guardar_nota`, `listar_notas` y `buscar_notas` | `uv run src/server.py` en `mi-servidor` |
| `fetch` | Lee páginas web | `uvx mcp-server-fetch` |
| `docs` | Consulta la documentación de LangChain (ver abajo) | `uv run src/server.py` en `docs-servidor` |

`.github/copilot-instructions.md` le pide a Copilot que consulte la memoria al empezar cada conversación y que guarde ahí los datos importantes.

Algunos puntos en los que tuvimos que apartarnos del tutorial:

- `uv add "mcp[cli]"` instala hoy la versión 2 del SDK, donde `FastMCP` ya no existe. Fijamos `mcp[cli]<2` para usar el código del tutorial tal cual.
- En el paso 8, `--directory` tiene que apuntar a `${workspaceFolder}/mi-servidor`, porque `src/server.py` está dentro de esa carpeta.
- El ejemplo del paso 8 deja en `mcp.json` solo el servidor de notas. Nosotros dejamos todos los servidores en el mismo archivo.
- En la versión actual de VS Code no aparece el botón Start sobre `mcp.json` hasta confiar en la carpeta. Si aun así no sale, se arrancan con `Cmd+Shift+P` y luego `MCP: List Servers`.

### Servidor docs

Es un servidor MCP en Python que responde preguntas consultando la documentación oficial de LangChain, LangGraph y LangSmith en el momento de la pregunta.

Funciona así:

1. LangChain publica un índice para asistentes de IA en `https://docs.langchain.com/llms.txt`. El servidor lo recorre junto con sus sub-índices y arma una lista de 1678 páginas con título, descripción y sección. La lista se guarda en caché 24 horas.
2. Para buscar usa BM25 sobre el título, la descripción, la sección y la ruta de cada página. El título y la descripción pesan más. Se normalizan los plurales y se filtran las páginas por lenguaje (Python por defecto).
3. Descarga en Markdown las páginas que mejor coinciden y devuelve solo las secciones que más se relacionan con la pregunta, con su URL para citarlas.

Herramientas que expone:

| Herramienta | Qué hace |
|---|---|
| `consultar_documentacion` | Busca, descarga y devuelve los fragmentos relevantes. Es la principal |
| `buscar_documentacion` | Lista las páginas que coinciden con una consulta |
| `leer_pagina` | Descarga una página completa |
| `actualizar_indice` | Vuelve a descargar el índice |

La documentación está en inglés, así que las herramientas piden la consulta en inglés y Copilot se encarga de traducir la pregunta. Por seguridad solo se leen páginas del dominio de la documentación.

El servidor sirve para cualquier librería que publique un `llms.txt`. Basta con cambiar `DOCS_LLMS_TXT` en `mcp.json`. Lo probamos también con la documentación de Pydantic (`https://docs.pydantic.dev/latest/llms.txt`).

La búsqueda es por palabras clave y no por significado, así que a veces la página más útil no queda entre las primeras. En la práctica Copilot hace varias búsquedas y lee las páginas que necesita.

### Cómo usarlo

```bash
cd proyecto-mcp
(cd mi-servidor && uv sync)
(cd docs-servidor && uv sync)
code .
```

Es importante abrir VS Code en `proyecto-mcp`, porque `mcp.json` usa `${workspaceFolder}`. Después:

1. Confiar en la carpeta si VS Code la abre en Restricted Mode.
2. `Cmd+Shift+P`, luego `MCP: List Servers`, y arrancar cada servidor.
3. En el chat de Copilot, en modo Agent, probar cosas como:
   - "Usa el mcp memory. Recuerda que me llamo Ana y prefiero ejemplos en Python."
   - "Guarda una nota titulada Ideas con el contenido Aprender MCP con uv."
   - "Usa el mcp docs: ¿cómo creo un agente con herramientas usando create_agent en LangChain?"

Para probar un servidor sin Copilot se puede usar el MCP Inspector:

```bash
cd docs-servidor
uv run mcp dev src/server.py
```
