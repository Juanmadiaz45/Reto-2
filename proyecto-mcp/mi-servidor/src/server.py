import json
from datetime import datetime
from pathlib import Path

# FastMCP es la forma más sencilla de crear un servidor con el SDK oficial
from mcp.server.fastmcp import FastMCP

# Nombre con el que el servidor se presenta al cliente
mcp = FastMCP("notas")

# Las notas se guardan en notas.json, en la misma carpeta que este script
ARCHIVO = Path(__file__).parent / "notas.json"

# Lee el archivo de notas; si todavía no existe, devuelve una lista vacía
def cargar() -> list[dict]:
    if ARCHIVO.exists():
        return json.loads(ARCHIVO.read_text(encoding="utf-8"))
    return []

# Escribe la lista completa; ensure_ascii=False conserva tildes y eñes
def guardar(notas: list[dict]) -> None:
    ARCHIVO.write_text(
        json.dumps(notas, ensure_ascii=False, indent=2), encoding="utf-8"
    )

# ---------- Herramientas que verá el asistente ----------
@mcp.tool()
def guardar_nota(titulo: str, contenido: str) -> str:
    """Guarda una nota con título y contenido."""
    notas = cargar()
    notas.append({
        "titulo": titulo,
        "contenido": contenido,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    })
    guardar(notas)
    return f"Nota '{titulo}' guardada. Total de notas: {len(notas)}"

@mcp.tool()
def listar_notas() -> list[dict]:
    """Devuelve todas las notas guardadas."""
    return cargar()

@mcp.tool()
def buscar_notas(texto: str) -> list[dict]:
    """Busca notas cuyo título o contenido contenga el texto indicado."""
    t = texto.lower()
    return [
        n for n in cargar()
        if t in n["titulo"].lower() or t in n["contenido"].lower()
    ]

# Punto de entrada: VS Code ejecuta este archivo y se comunica con él por stdio
if __name__ == "__main__":
    mcp.run()  # usa stdio por defecto
