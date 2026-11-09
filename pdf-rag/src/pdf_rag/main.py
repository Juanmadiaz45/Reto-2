"""Interfaz de consola: indexar un PDF, preguntar, chatear y evaluar."""
import argparse
import json

from .config import BASE_DIR, langsmith_activo


def _imprimir_resumen(r: dict) -> None:
    print("\n=== Precisión (RAGAS, 0 a 1) ===")
    for metrica, valor in r["ragas_promedios"].items():
        print(f"  {metrica:<45} {valor:.3f}")
    c = r["costo_rag_local"]
    print("\n=== Precio del RAG ===")
    print(f"  Llamadas al LLM: {c['llamadas']}  |  tokens entrada/salida: "
          f"{c['tokens_entrada']}/{c['tokens_salida']}")
    print(f"  Costo total: ${c['costo_usd']:.6f}  |  por pregunta: ${r['costo_rag_por_pregunta_usd']:.6f}")
    ls = r["costo_rag_langsmith"]
    if ls and "error" in ls:
        print(f"  LangSmith: {ls['error']}")
    elif ls:
        print(f"  Según LangSmith: ${ls['costo_total_usd']:.6f}  (${ls['costo_por_traza_usd']:.6f} por pregunta, "
              f"{ls['tokens_totales']} tokens, latencia p50 {ls['latencia_p50_s']} s)")
    else:
        print("  LangSmith desactivado (defina LANGSMITH_API_KEY y LANGSMITH_TRACING=true en .env)")
    j = r["costo_evaluacion_ragas"]
    print(f"\n  Costo de la evaluación con RAGAS (juez): ${j['costo_usd']:.6f}")
    print(f"\nResultados guardados en resultados/{r['experimento']}.csv y .json")


def main():
    parser = argparse.ArgumentParser(prog="pdf-rag", description="RAG sobre PDFs con ChromaDB, RAGAS y LangSmith")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("indexar", help="Lee un PDF y lo guarda en ChromaDB")
    p.add_argument("pdf")

    p = sub.add_parser("preguntar", help="Hace una pregunta sobre los PDFs indexados")
    p.add_argument("pregunta")
    p.add_argument("-k", type=int, default=4, help="Fragmentos a recuperar")

    sub.add_parser("chat", help="Chat interactivo sobre los PDFs indexados")

    p = sub.add_parser("generar-dataset", help="Crea preguntas de prueba a partir del PDF")
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--salida", default=str(BASE_DIR / "datos" / "dataset.json"))

    p = sub.add_parser("evaluar", help="Mide precisión (RAGAS) y precio (LangSmith)")
    p.add_argument("--dataset", default=str(BASE_DIR / "datos" / "dataset.json"))
    p.add_argument("-k", type=int, default=4)

    p = sub.add_parser("costo", help="Consulta en LangSmith el costo acumulado del proyecto")
    p.add_argument("--experimento", default=None)

    args = parser.parse_args()

    if args.comando == "costo":
        from .evaluacion import costo_langsmith
        if not langsmith_activo():
            print("LangSmith desactivado (defina LANGSMITH_API_KEY y LANGSMITH_TRACING=true en .env)")
            return
        print(json.dumps(costo_langsmith(args.experimento), ensure_ascii=False, indent=2))
        return

    from .rag import RagPDF
    rag = RagPDF()

    if args.comando == "indexar":
        n = rag.indexar(args.pdf)
        print(f"Indexados {n} fragmentos de {args.pdf}. Total en la colección: {rag.coleccion.count()}")

    elif args.comando == "preguntar":
        print(rag.responder(args.pregunta, args.k)["respuesta"])

    elif args.comando == "chat":
        print("Pregunte sobre el PDF. Escriba 'salir' para terminar.\n")
        while True:
            texto = input("Usted: ").strip()
            if texto.lower() in {"salir", "exit"}:
                break
            if texto:
                print(f"Asistente: {rag.responder(texto)['respuesta']}\n")

    elif args.comando == "generar-dataset":
        from .evaluacion import generar_dataset
        datos = generar_dataset(rag, args.n, args.salida)
        print(f"{len(datos)} preguntas guardadas en {args.salida}. Revíselas antes de evaluar.")

    elif args.comando == "evaluar":
        from .evaluacion import cargar_dataset, evaluar
        dataset = cargar_dataset(args.dataset)
        print(f"Evaluando {len(dataset)} preguntas...")
        _imprimir_resumen(evaluar(rag, dataset, args.k))


if __name__ == "__main__":
    main()
