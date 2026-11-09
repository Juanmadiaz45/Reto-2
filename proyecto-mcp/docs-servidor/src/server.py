import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

# Nombre con el que el servidor se presenta al cliente
mcp = FastMCP("docs")

# Índice llms.txt de la documentación a consultar (LangChain por defecto).
# Cualquier librería que publique un llms.txt sirve: basta con cambiar DOCS_LLMS_TXT en mcp.json.
LLMS_TXT = os.environ.get("DOCS_LLMS_TXT", "https://docs.langchain.com/llms.txt")
DOMINIO = urlparse(LLMS_TXT).netloc
# Dominios desde los que se aceptan páginas: el del índice y aquel al que redirija
DOMINIOS = {DOMINIO}

# El índice se guarda en disco para no descargarlo en cada arranque
CACHE = Path(__file__).parent / f"indice-{DOMINIO}.json"
VIGENCIA_CACHE = 24 * 3600  # segundos

# Líneas del índice: "- [Título](url): descripción"
ENLACE = re.compile(r"^\s*[-*]\s*\[([^\]]+)\]\(([^)\s]+)\)(?::\s*(.*))?$")
TITULO = re.compile(r"^(#{2,6})\s+(.*)$")
PALABRA = re.compile(r"[a-z0-9áéíóúñü]+")

# Marcas con las que la documentación separa las versiones por lenguaje
LENGUAJES = {"python": ("/python/", "python"), "typescript": ("/javascript/", "typescript")}

# Sin esto, cada petición HTTP aparece como un aviso en el Output de VS Code
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------- Utilidades ----------
def log(*args) -> None:
    # Nunca print() a stdout: en un servidor stdio rompe el protocolo
    print(*args, file=sys.stderr)


def es_indice(url: str) -> bool:
    """Los sub-índices son .txt (por ejemplo /_llms/python.txt); el resto son páginas."""
    ruta = urlparse(url).path
    return ruta.endswith(".txt") and "llms" in ruta


def tokenizar(texto: str) -> list[str]:
    # Singular aproximado: "agents" y "agent", "validators" y "validator" cuentan igual
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in PALABRA.findall(texto.lower())]


def cliente_http() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=30,
                             headers={"User-Agent": "docs-mcp/0.1 (tutorial MCP)"})


# ---------- Construcción del índice ----------
async def descargar_indice() -> list[dict]:
    """Recorre llms.txt y sus sub-índices y devuelve las páginas con título, url, descripción y sección."""
    paginas: dict[str, dict] = {}
    visitados: set[str] = set()
    pendientes = [LLMS_TXT]

    async with cliente_http() as http:
        while pendientes:
            lote, pendientes = pendientes[:8], pendientes[8:]
            visitados.update(lote)
            respuestas = await asyncio.gather(*(http.get(u) for u in lote), return_exceptions=True)
            for url_indice, r in zip(lote, respuestas):
                if isinstance(r, Exception) or r.status_code != 200:
                    log(f"No se pudo leer {url_indice}: {r}")
                    continue
                if url_indice == LLMS_TXT:  # p. ej. docs.pydantic.dev redirige a pydantic.dev
                    DOMINIOS.add(r.url.host)
                url_indice = str(r.url)
                secciones: list[str] = []
                for linea in r.text.splitlines():
                    if m := TITULO.match(linea):
                        nivel = len(m.group(1)) - 2
                        secciones = secciones[:nivel] + [m.group(2).strip()]
                        continue
                    if not (m := ENLACE.match(linea)):
                        continue
                    titulo, url, descripcion = m.group(1), urljoin(url_indice, m.group(2)), m.group(3) or ""
                    if urlparse(url).netloc not in DOMINIOS:
                        continue
                    if es_indice(url):
                        if url not in visitados and url not in pendientes:
                            pendientes.append(url)
                        continue
                    seccion = " > ".join(secciones)
                    if url in paginas:  # la misma página puede aparecer en varios índices
                        if seccion and seccion not in paginas[url]["seccion"]:
                            paginas[url]["seccion"] += f" | {seccion}"
                    else:
                        paginas[url] = {"titulo": titulo, "url": url,
                                        "descripcion": descripcion.strip(), "seccion": seccion}
    log(f"Índice de {DOMINIO}: {len(paginas)} páginas ({len(visitados)} índices leídos)")
    return list(paginas.values())


class Indice:
    """Búsqueda BM25 sobre título, descripción, sección y ruta de cada página."""

    def __init__(self, paginas: list[dict]):
        self.paginas = paginas
        self.docs = []
        for p in paginas:
            ruta = urlparse(p["url"]).path.replace("/", " ").replace("-", " ")
            # El título y la descripción pesan más que la sección y la ruta
            texto = f"{p['titulo']} {p['titulo']} {p['titulo']} {p['descripcion']} {p['descripcion']} {p['seccion']} {ruta}"
            self.docs.append(Counter(tokenizar(texto)))
        self.largo_medio = sum(sum(d.values()) for d in self.docs) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def buscar(self, consulta: str, k: int, lenguaje: str = "") -> list[dict]:
        terminos = tokenizar(consulta)
        otros = [m for l, m in LENGUAJES.items() if lenguaje and l != lenguaje.lower()]
        puntuados = []
        for p, d in zip(self.paginas, self.docs):
            # Descarta las páginas de otro lenguaje (p. ej. JavaScript si se pidió Python)
            if any(ruta in p["url"] or p["seccion"].lower().startswith(sec) for ruta, sec in otros):
                continue
            largo = sum(d.values())
            s = 0.0
            for t in terminos:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * largo / self.largo_medio))
            if s > 0:
                puntuados.append((s, p))
        puntuados.sort(key=lambda x: x[0], reverse=True)
        return [{**p, "puntuacion": round(s, 2)} for s, p in puntuados[:k]]


_indice: Indice | None = None


async def obtener_indice(forzar: bool = False) -> Indice:
    global _indice
    if _indice is not None and not forzar:
        return _indice
    if not forzar and CACHE.exists() and time.time() - CACHE.stat().st_mtime < VIGENCIA_CACHE:
        datos = json.loads(CACHE.read_text(encoding="utf-8"))
        paginas = datos["paginas"]
        DOMINIOS.update(datos["dominios"])
    else:
        paginas = await descargar_indice()
        CACHE.write_text(json.dumps({"dominios": sorted(DOMINIOS), "paginas": paginas},
                                    ensure_ascii=False), encoding="utf-8")
    _indice = Indice(paginas)
    return _indice


# ---------- Lectura de páginas ----------
async def descargar_pagina(url: str) -> str:
    await obtener_indice()  # carga los dominios permitidos
    if urlparse(url).netloc not in DOMINIOS:
        raise ValueError(f"Solo se pueden leer páginas de {', '.join(sorted(DOMINIOS))}")
    async with cliente_http() as http:
        r = await http.get(url)
        r.raise_for_status()
    texto = r.text
    # Quita el aviso "Documentation Index" que algunas páginas traen al principio
    lineas = texto.splitlines()
    while lineas and (lineas[0].startswith(">") or not lineas[0].strip()):
        lineas.pop(0)
    return "\n".join(lineas)


def fragmentos_relevantes(texto: str, consulta: str, max_caracteres: int) -> str:
    """Se queda con los bloques (separados por títulos) que más términos de la consulta contienen."""
    bloques = re.split(r"\n(?=#{1,4} )", texto)
    terminos = set(tokenizar(consulta))
    puntuados = []
    for i, b in enumerate(bloques):
        palabras = Counter(tokenizar(b))
        puntuados.append((sum(palabras[t] for t in terminos), i, b))
    elegidos, total = [], 0
    for s, i, b in sorted(puntuados, key=lambda x: (-x[0], x[1])):
        if total + len(b) > max_caracteres and elegidos:
            continue
        elegidos.append((i, b[:max_caracteres]))
        total += len(b)
    return "\n\n".join(b for _, b in sorted(elegidos))  # en el orden original de la página


# ---------- Herramientas que verá el asistente ----------
@mcp.tool()
async def buscar_documentacion(consulta: str, max_resultados: int = 8,
                               lenguaje: str = "python") -> list[dict]:
    """Busca páginas en la documentación oficial (por defecto LangChain, LangGraph y LangSmith).

    Escribe la consulta EN INGLÉS y con términos técnicos (por ejemplo "create_agent tools
    middleware", "checkpointer memory langgraph python"), porque la documentación está en inglés.
    lenguaje: "python", "typescript" o "" para no filtrar.
    Devuelve título, url, descripción y sección de cada página; usa leer_pagina para ver el contenido.
    """
    indice = await obtener_indice()
    return indice.buscar(consulta, max_resultados, lenguaje)


@mcp.tool()
async def leer_pagina(url: str, max_caracteres: int = 15000) -> str:
    """Descarga una página de la documentación (en Markdown) a partir de su url."""
    texto = await descargar_pagina(url)
    if len(texto) > max_caracteres:
        texto = texto[:max_caracteres] + f"\n\n[... página recortada: {len(texto)} caracteres en total]"
    return texto


@mcp.tool()
async def consultar_documentacion(pregunta: str, consulta_en_ingles: str, paginas: int = 3,
                                  lenguaje: str = "python") -> str:
    """Responde una pregunta consultando la documentación en vivo: busca las páginas más
    relevantes, las descarga y devuelve los fragmentos útiles con su url para citarlas.

    Úsala primero ante cualquier pregunta sobre la librería documentada.
    - pregunta: la pregunta del usuario, tal cual.
    - consulta_en_ingles: palabras clave en inglés para buscar (la documentación está en inglés).
    - paginas: cuántas páginas leer (1 a 5).
    - lenguaje: "python", "typescript" o "" para no filtrar.
    """
    indice = await obtener_indice()
    resultados = indice.buscar(consulta_en_ingles, max(1, min(paginas, 5)), lenguaje)
    if not resultados:
        return "No encontré páginas relevantes. Prueba otras palabras clave en inglés con buscar_documentacion."

    textos = await asyncio.gather(*(descargar_pagina(r["url"]) for r in resultados),
                                  return_exceptions=True)
    partes = [f"Pregunta: {pregunta}\nFuente: {LLMS_TXT}\n"]
    for r, texto in zip(resultados, textos):
        if isinstance(texto, Exception):
            partes.append(f"## {r['titulo']}\nURL: {r['url']}\n(No se pudo descargar: {texto})")
            continue
        extracto = fragmentos_relevantes(texto, f"{consulta_en_ingles} {pregunta}", 6000)
        partes.append(f"## {r['titulo']}\nURL: {r['url']}\nSección: {r['seccion']}\n\n{extracto}")
    return "\n\n---\n\n".join(partes)


@mcp.tool()
async def actualizar_indice() -> str:
    """Vuelve a descargar el índice de la documentación (útil si la documentación cambió hoy)."""
    indice = await obtener_indice(forzar=True)
    return f"Índice actualizado: {len(indice.paginas)} páginas de {DOMINIO}."


# Punto de entrada: VS Code ejecuta este archivo y se comunica con él por stdio
if __name__ == "__main__":
    mcp.run()  # usa stdio por defecto
