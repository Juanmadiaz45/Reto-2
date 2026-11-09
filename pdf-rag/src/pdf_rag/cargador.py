"""Lectura del PDF, división en fragmentos e indexación en ChromaDB."""
import hashlib
from pathlib import Path

from langsmith import traceable
from pypdf import PdfReader


def leer_pdf(ruta: str) -> list[dict]:
    """Extrae el texto de cada página: [{"pagina": 1, "texto": "..."}]."""
    lector = PdfReader(ruta)
    paginas = []
    for numero, pagina in enumerate(lector.pages, start=1):
        texto = " ".join((pagina.extract_text() or "").split())  # normaliza espacios
        if texto:
            paginas.append({"pagina": numero, "texto": texto})
    return paginas


def dividir_en_fragmentos(paginas: list[dict], tamano: int = 1000, solape: int = 200) -> list[dict]:
    """Corta cada página en fragmentos con solape, intentando no partir palabras."""
    fragmentos = []
    for p in paginas:
        texto, inicio = p["texto"], 0
        while inicio < len(texto):
            fin = min(inicio + tamano, len(texto))
            if fin < len(texto):
                espacio = texto.rfind(" ", inicio, fin)
                if espacio > inicio + tamano // 2:
                    fin = espacio
            fragmentos.append({"texto": texto[inicio:fin].strip(), "pagina": p["pagina"]})
            if fin >= len(texto):
                break
            inicio = max(fin - solape, inicio + 1)
    return [f for f in fragmentos if f["texto"]]


@traceable(run_type="chain", name="indexar_pdf")
def indexar_pdf(coleccion, ruta: str, tamano: int = 1000, solape: int = 200) -> int:
    """Lee el PDF y guarda sus fragmentos en la colección. Devuelve cuántos guardó.

    Los ids se derivan del contenido, así que reindexar el mismo PDF no duplica datos.
    """
    nombre = Path(ruta).name
    fragmentos = dividir_en_fragmentos(leer_pdf(ruta), tamano, solape)
    if not fragmentos:
        raise ValueError(f"No se pudo extraer texto de {nombre} (¿es un PDF escaneado?).")

    ids, documentos, metadatos = [], [], []
    for i, f in enumerate(fragmentos):
        huella = hashlib.sha1(f"{nombre}|{i}|{f['texto']}".encode()).hexdigest()[:16]
        ids.append(f"{nombre}-{huella}")
        documentos.append(f["texto"])
        metadatos.append({"fuente": nombre, "pagina": f["pagina"], "fragmento": i})

    lote = 100
    for i in range(0, len(ids), lote):
        coleccion.upsert(ids=ids[i:i + lote], documents=documentos[i:i + lote],
                         metadatas=metadatos[i:i + lote])
    return len(ids)
