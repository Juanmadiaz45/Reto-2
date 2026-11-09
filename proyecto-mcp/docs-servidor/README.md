# docs-servidor

Servidor MCP que consulta en vivo la documentación de LangChain a partir de su `llms.txt`. Expone `consultar_documentacion`, `buscar_documentacion`, `leer_pagina` y `actualizar_indice`. La documentación a consultar se cambia con la variable `DOCS_LLMS_TXT`.

```bash
uv sync
uv run mcp dev src/server.py   # probarlo con el MCP Inspector
```
