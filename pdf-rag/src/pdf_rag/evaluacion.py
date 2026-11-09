"""Evaluación del RAG: precisión con RAGAS y precio con LangSmith."""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from .config import (MODELO, MODELO_EMBEDDINGS, MODELO_EVALUADOR, PRECIOS, PROYECTO_LANGSMITH,
                     RUTA_RESULTADOS, langsmith_activo)
from .llm import contador, llamar_llm
from .rag import RagPDF

PROMPT_DATASET = (
    "A partir del fragmento de un documento, escriba UNA pregunta que se pueda responder solo "
    "con ese fragmento y su respuesta correcta y completa, redactada en 1 o 2 oraciones "
    "(no copie el fragmento). Responda SOLO con un JSON válido: "
    '{"pregunta": "...", "referencia": "..."}'
)


# ---------------------------------------------------------------- dataset de prueba

def generar_dataset(rag: RagPDF, n: int, ruta: str) -> list[dict]:
    """Crea n pares pregunta/respuesta de referencia a partir de fragmentos del PDF."""
    total = rag.coleccion.count()
    if total == 0:
        raise RuntimeError("No hay documentos indexados.")
    datos = rag.coleccion.get(include=["documents", "metadatas"])
    paso = max(total // n, 1)
    dataset = []
    for i in range(0, total, paso)[:n]:
        salida = llamar_llm(PROMPT_DATASET, [{"role": "user", "content": datos["documents"][i]}])
        try:
            par = json.loads(salida.replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError:
            continue
        if par.get("pregunta") and par.get("referencia"):
            dataset.append({"pregunta": par["pregunta"], "referencia": par["referencia"],
                            "pagina": datos["metadatas"][i]["pagina"]})
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def cargar_dataset(ruta: str) -> list[dict]:
    """Lee un JSON con la forma [{"pregunta": "...", "referencia": "..."}]."""
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- RAGAS

def _metricas_ragas():
    from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    return [
        LLMContextPrecisionWithReference(),  # ¿los fragmentos recuperados son relevantes?
        LLMContextRecall(),                  # ¿se recuperó todo lo necesario?
        Faithfulness(),                      # ¿la respuesta se apoya en los fragmentos?
        ResponseRelevancy(),                 # ¿la respuesta atiende la pregunta?
        FactualCorrectness(),                # ¿la respuesta coincide con la referencia?
    ]


def _evaluador_ragas():
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # Groq no admite n > 1: RAGAS hará varias llamadas en lugar de pedir varias respuestas
    llm = LangchainLLMWrapper(ChatGroq(model=MODELO_EVALUADOR, temperature=0, max_tokens=8000),
                              bypass_n=True)
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS))
    return llm, emb


# ---------------------------------------------------------------- LangSmith

def _registrar_feedback(trazas: list, df) -> None:
    """Adjunta cada puntuación RAGAS a su traza en LangSmith (precisión + precio en un mismo lugar)."""
    from langsmith import Client

    cliente = Client()
    columnas = [c for c in df.columns if df[c].dtype.kind == "f"]
    for traza, (_, fila) in zip(trazas, df.iterrows()):
        for col in columnas:
            if fila[col] == fila[col]:  # descarta NaN
                cliente.create_feedback(run_id=traza.id, session_id=traza.session_id,
                                        key=f"ragas_{col}", score=float(fila[col]))


def _leer_trazas(run_ids: list[str] | None, experimento: str | None, esperadas: int,
                 intentos: int = 8) -> list:
    """Lee de LangSmith las trazas raíz; reintenta mientras la ingesta no haya terminado."""
    from langsmith import Client
    from langsmith.run_trees import get_cached_client
    from langsmith.utils import LangSmithRateLimitError

    get_cached_client().flush()  # envía las trazas pendientes antes de consultar
    filtro = None
    if experimento and not run_ids:
        filtro = f'and(eq(metadata_key, "experimento"), eq(metadata_value, "{experimento}"))'
    cliente, trazas, espera = Client(), [], 3
    for _ in range(intentos):
        try:
            trazas = list(cliente.list_runs(project_name=PROYECTO_LANGSMITH, is_root=True,
                                            run_ids=run_ids, filter=filtro))
            # LangSmith suma el costo de los hijos al cerrar la traza: esperar a que esté listo
            if len(trazas) >= esperadas and all(t.end_time and t.total_cost is not None
                                                for t in trazas):
                break
        except LangSmithRateLimitError:
            pass
        time.sleep(espera)
        espera = min(espera * 2, 30)
    return trazas


def costo_langsmith(experimento: str | None = None, run_ids: list[str] | None = None) -> dict | None:
    """Tokens, costo y latencia que LangSmith registró para un experimento (o todo el proyecto)."""
    if not langsmith_activo():
        return None
    trazas = _leer_trazas(run_ids, experimento, esperadas=len(run_ids) if run_ids else 1)
    if not trazas:
        return {"error": "LangSmith aún no devolvió las trazas; reintente con 'pdf-rag costo'."}
    latencias = sorted((t.end_time - t.start_time).total_seconds() for t in trazas if t.end_time)
    costo = sum(float(t.total_cost or 0) for t in trazas)
    return {
        "trazas": len(trazas),
        "tokens_entrada": sum(t.prompt_tokens or 0 for t in trazas),
        "tokens_salida": sum(t.completion_tokens or 0 for t in trazas),
        "tokens_totales": sum(t.total_tokens or 0 for t in trazas),
        "costo_total_usd": round(costo, 6),
        "costo_por_traza_usd": round(costo / len(trazas), 6),
        "latencia_p50_s": round(latencias[len(latencias) // 2], 2) if latencias else None,
        "latencia_max_s": round(latencias[-1], 2) if latencias else None,
    }


# ---------------------------------------------------------------- evaluación completa

def evaluar(rag: RagPDF, dataset: list[dict], k: int = 4) -> dict:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cost import get_token_usage_for_openai

    experimento = f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    contador.reiniciar()

    # 1) Ejecutar el RAG sobre cada pregunta, guardando el id de su traza en LangSmith
    muestras, ids_runs = [], []
    for i, item in enumerate(dataset, start=1):
        run_id = str(uuid.uuid4())
        salida = rag.responder(item["pregunta"], k, langsmith_extra={
            "run_id": run_id, "metadata": {"experimento": experimento}})
        ids_runs.append(run_id)
        muestras.append({
            "user_input": item["pregunta"],
            "retrieved_contexts": salida["contextos"],
            "response": salida["respuesta"],
            "reference": item["referencia"],
        })
        print(f"  [{i}/{len(dataset)}] {item['pregunta'][:70]}")
    costo_rag = {"llamadas": contador.llamadas, "tokens_entrada": contador.tokens_entrada,
                 "tokens_salida": contador.tokens_salida, "costo_usd": round(contador.costo, 6)}

    # 2) Medir la precisión con RAGAS (un LLM de Groq como juez)
    llm, emb = _evaluador_ragas()
    resultado = evaluate(
        EvaluationDataset.from_list(muestras),
        metrics=_metricas_ragas(),
        llm=llm,
        embeddings=emb,
        experiment_name=experimento,
        token_usage_parser=get_token_usage_for_openai,  # Groq usa el formato de OpenAI
        # Pocas peticiones en paralelo y reintentos largos: Groq limita peticiones por minuto
        run_config=RunConfig(max_workers=2, timeout=300, max_retries=10, max_wait=60),
    )
    df = resultado.to_pandas()
    columnas = [c for c in df.columns if df[c].dtype.kind == "f"]
    promedios = {c: round(float(df[c].mean()), 4) for c in columnas}

    p_entrada, p_salida = PRECIOS.get(MODELO_EVALUADOR, (0.0, 0.0))
    uso_juez = resultado.total_tokens()
    costo_juez = {
        "tokens_entrada": uso_juez.input_tokens,
        "tokens_salida": uso_juez.output_tokens,
        "costo_usd": round(resultado.total_cost(cost_per_input_token=p_entrada / 1e6,
                                                cost_per_output_token=p_salida / 1e6), 6),
    }

    # 3) Precio según LangSmith + puntuaciones como feedback en cada traza
    costo_ls = None
    if langsmith_activo():
        try:
            costo_ls = costo_langsmith(experimento, ids_runs)
            trazas = {str(t.id): t for t in _leer_trazas(ids_runs, None, len(ids_runs), intentos=1)}
            _registrar_feedback([trazas[i] for i in ids_runs if i in trazas], df)
        except Exception as e:  # un fallo de LangSmith no debe perder los resultados de RAGAS
            costo_ls = {"error": f"{type(e).__name__}: {e}"}

    resumen = {
        "experimento": experimento,
        "modelo_rag": MODELO,
        "modelo_evaluador": MODELO_EVALUADOR,
        "preguntas": len(dataset),
        "k": k,
        "ragas_promedios": promedios,
        "costo_rag_local": costo_rag,
        "costo_rag_por_pregunta_usd": round(costo_rag["costo_usd"] / max(len(dataset), 1), 6),
        "costo_rag_langsmith": costo_ls,
        "costo_evaluacion_ragas": costo_juez,
    }

    RUTA_RESULTADOS.mkdir(exist_ok=True)
    df.to_csv(RUTA_RESULTADOS / f"{experimento}.csv", index=False)
    (RUTA_RESULTADOS / f"{experimento}.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen
