"""Llamadas al LLM con trazas en LangSmith que incluyen tokens y precio."""
import os
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from .config import MODELO, calcular_costo


class ContadorCosto:
    """Acumula tokens y costo de todas las llamadas del proceso (respaldo local de LangSmith)."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.llamadas = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0
        self.costo = 0.0

    def sumar(self, tokens_entrada: int, tokens_salida: int, costo: float):
        self.llamadas += 1
        self.tokens_entrada += tokens_entrada
        self.tokens_salida += tokens_salida
        self.costo += costo


contador = ContadorCosto()


@traceable(run_type="llm", name="groq", metadata={"ls_provider": "groq"})
def llamar_llm(sistema: str, mensajes: list[dict[str, Any]], max_tokens: int = 2048,
               modelo: str = MODELO) -> str:
    """Envía un prompt de sistema y una lista de mensajes al LLM y devuelve el texto."""
    if not mensajes:
        raise ValueError("La lista de mensajes no puede estar vacía.")

    if not os.environ.get("GROQ_API_KEY"):
        # Modo de prueba: sin clave de API devuelve una respuesta simulada
        return "[respuesta simulada] " + str(mensajes[-1].get("content", ""))[:200]

    from groq import Groq

    extra = {}
    if "gpt-oss" in modelo:  # modelos de razonamiento: poco esfuerzo y sin devolver el razonamiento
        extra = {"reasoning_effort": "low", "include_reasoning": False}

    cliente = Groq()
    respuesta = cliente.chat.completions.create(
        model=modelo,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": sistema}, *mensajes],
        temperature=0,
        **extra,
    )

    uso = respuesta.usage
    costo_entrada, costo_salida = calcular_costo(modelo, uso.prompt_tokens, uso.completion_tokens)
    contador.sumar(uso.prompt_tokens, uso.completion_tokens, costo_entrada + costo_salida)

    # LangSmith lee usage_metadata para mostrar tokens y precio de cada llamada
    traza = get_current_run_tree()
    if traza is not None:
        traza.set(metadata={"ls_model_name": modelo}, usage_metadata={
            "input_tokens": uso.prompt_tokens,
            "output_tokens": uso.completion_tokens,
            "total_tokens": uso.prompt_tokens + uso.completion_tokens,
            "input_cost": costo_entrada,
            "output_cost": costo_salida,
            "total_cost": costo_entrada + costo_salida,
        })

    return respuesta.choices[0].message.content or ""
