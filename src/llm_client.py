import os
import json
import time
import difflib
import hashlib
from collections.abc import Callable
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        # OPENROUTER_MODEL puede ser una lista separada por comas para rotación
        # Ejemplo: "openai/gpt-oss-120b:free, qwen/qwen3-coder:free"
        models_str = os.getenv(
            "OPENROUTER_MODEL",
            "openai/gpt-oss-120b:free, qwen/qwen3-coder:free"
        )
        self.model_list: list[str] = [m.strip() for m in models_str.split(",") if m.strip()]
        self.model_name: str = self.model_list[0]  # modelo activo (para logs)
        # Caché para evitar múltiples llamadas a la API con los mismos inputs
        self._cache: dict[str, dict] = {
            "parse_user_goal": {},
            "score_resources": {},
            "compare_algorithms": {},
            "explain_path": {}
        }

    def _make_cache_key(self, data: str | list | dict | set | tuple) -> str:
        """Genera una clave de caché usando hash SHA256 del input."""
        if isinstance(data, (list, dict, set)):
            data = json.dumps(data, sort_keys=True, default=str)
        elif not isinstance(data, str):
            data = str(data)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _call_llm(self, prompt: str) -> str:
        """
        Llamada base al LLM con rotación automática de modelos ante rate limit (429).

        Estrategia: si el modelo activo devuelve 429 por límite del proveedor
        (Venice, Together, etc.), rotar al siguiente de model_list SIN esperar —
        el rate limit de proveedor no se resuelve esperando, solo cambiando de modelo.
        Solo espera si todos los modelos de la lista fallan (caso extremo).
        """
        last_error = None
        for model in self.model_list:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                if model != self.model_name:
                    print(f"\n[Info] Respondió: {model}")
                return response.choices[0].message.content.strip()
            except RateLimitError as e:
                last_error = e
                idx = self.model_list.index(model)
                if idx < len(self.model_list) - 1:
                    print(f"\n[Rate limit en {model}] "
                          f"Rotando a {self.model_list[idx + 1]}...")
                # si es el último, salimos del bucle y gestionamos abajo

        # Todos los modelos fallaron: esperar y hacer un último intento
        retry_after = 25.0
        try:
            body = (last_error.body or {}) if last_error else {}
            retry_after = float(
                body.get("error", {})
                    .get("metadata", {})
                    .get("retry_after_seconds", 25)
            )
        except Exception:
            pass
        wait = round(retry_after + 2, 1)
        print(f"\n[Rate limit en todos los modelos] Esperando {wait}s y reintentando...")
        time.sleep(wait)
        response = self.client.chat.completions.create(
            model=self.model_list[0],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _snap_skills(skills: list, available_skills: list[str]) -> list[str]:
        """Sanea las skills que devuelve el LLM contra el catálogo real.

        El LLM free-tier a veces devuelve un nombre *casi* correcto que no está en
        la lista permitida (p.ej. `api_rest` por `apis_rest`, singular/plural). Sin
        este saneo, `check_feasibility` lo marca como "fuera del catálogo" pese a
        existir el recurso. Estrategia: exacto -> se mantiene; si no, se "ajusta" al
        más parecido del catálogo (difflib, umbral alto); si no hay parecido (skill
        de otro dominio), se descarta. Preserva el orden y elimina duplicados.
        """
        out: list[str] = []
        for s in skills:
            if not isinstance(s, str):
                continue
            if s in available_skills:
                out.append(s)
                continue
            match = difflib.get_close_matches(s, available_skills, n=1, cutoff=0.8)
            if match:
                out.append(match[0])
            # sin match cercano -> se descarta (fuera de dominio)
        seen: set[str] = set()
        return [s for s in out if not (s in seen or seen.add(s))]

    def parse_user_goal(self, user_input: str, available_skills: list[str]) -> dict:
        """Extrae habilidades objetivo, previas y límite de horas del input del usuario."""
        fallback = {
            "target_skills": [],
            "known_skills": [],
            "max_hours": None,
            "goal_summary": user_input
        }

        # Verificar caché
        cache_key = self._make_cache_key((user_input, available_skills))
        if cache_key in self._cache["parse_user_goal"]:
            return self._cache["parse_user_goal"][cache_key]

        try:
            skills_str = ", ".join(available_skills)
            prompt = (
                "Eres un extractor de información estructurada. "
                "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                "sin comillas de código, sin explicaciones.\n\n"
                f"El usuario quiere generar una ruta de aprendizaje personalizada.\n"
                f"Su objetivo en sus propias palabras es: \"{user_input}\"\n\n"
                f"IMPORTANTE: Solo puedes usar habilidades de esta lista exacta:\n{skills_str}\n\n"
                "Extrae la siguiente información y responde ÚNICAMENTE con un JSON válido:\n\n"
                "{\n"
                "  \"target_skills\": [\"habilidades de la lista que mejor representan lo que quiere aprender\"],\n"
                "  \"known_skills\": [\"habilidades de la lista que ya menciona tener, vacío si no menciona ninguna\"],\n"
                "  \"max_hours\": null,\n"
                "  \"goal_summary\": \"resumen del objetivo en una oración\"\n"
                "}\n\n"
                "Reglas estrictas:\n"
                "- Usa ÚNICAMENTE habilidades que aparezcan en la lista proporcionada\n"
                "- Si el usuario menciona \"machine learning\", mapéalo a las habilidades relevantes de la lista\n"
                "- Si el usuario dice \"no sé nada\", known_skills debe ser una lista vacía\n"
                "- Selecciona todas las habilidades relevantes al objetivo, no solo una"
            )

            raw = self._call_llm(prompt)
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            # Validar que las claves esperadas existen
            for key in ["target_skills", "known_skills", "max_hours", "goal_summary"]:
                if key not in result:
                    result[key] = fallback[key]

            # Sanear las skills contra el catálogo real (corrige near-misses del LLM
            # como api_rest -> apis_rest; descarta lo que sea de otro dominio).
            result["target_skills"] = self._snap_skills(result["target_skills"], available_skills)
            result["known_skills"] = self._snap_skills(result["known_skills"], available_skills)

            self._cache["parse_user_goal"][cache_key] = result
            return result

        except json.JSONDecodeError as e:
            print(f"\n[Advertencia] El LLM devolvió una respuesta no válida al parsear objetivo: {e}")
            print("Continuando con valores por defecto...")
            return fallback
        except Exception as e:
            print(f"\n[Advertencia] Error al conectar con el LLM: {e}")
            print("Continuando con valores por defecto...")
            return fallback

    def explain_path(self, path_result: dict, goal_summary: str,
                     get_resource_fn: Callable[[str], dict | None]) -> str:
        """Genera una explicación motivadora de la ruta recomendada."""
        fallback = (
            f"Tu ruta de aprendizaje cubre el {path_result['coverage_pct']}% de tu objetivo "
            f"en {path_result['total_hours']} horas con {len(path_result['path'])} recursos. "
            "Sigue la secuencia propuesta respetando el orden, ya que cada recurso "
            "construye sobre el anterior."
        )

        # Verificar caché
        cache_key = self._make_cache_key((
            goal_summary,
            tuple(path_result["path"]),
            path_result["algorithm"],
            path_result["coverage_pct"]
        ))
        if cache_key in self._cache["explain_path"]:
            return self._cache["explain_path"][cache_key]

        try:
            resources_detail = []
            for rid in path_result["path"]:
                r = get_resource_fn(rid)
                resources_detail.append(
                    f"- {r['name']} ({r['duration_hours']}h, {r['type']}): "
                    f"enseña {', '.join(r['teaches'])}"
                )
            resources_str = "\n".join(resources_detail)

            prompt = (
                "Eres un tutor experto en aprendizaje personalizado.\n\n"
                f"El usuario quiere: {goal_summary}\n\n"
                f"El sistema generó esta ruta de aprendizaje usando el algoritmo {path_result['algorithm']}:\n\n"
                f"{resources_str}\n\n"
                f"Duración total: {path_result['total_hours']} horas\n"
                f"Cobertura del objetivo: {path_result['coverage_pct']}%\n\n"
                "Escribe una explicación motivadora y clara para el usuario. Debe incluir:\n"
                "1. Por qué esta secuencia tiene sentido (menciona las dependencias clave)\n"
                "2. Qué va a lograr al final\n"
                "3. Un consejo práctico para mantener el ritmo\n\n"
                "Máximo 200 palabras. Habla directamente al usuario (usa \"tú\")."
            )

            result = self._call_llm(prompt)
            self._cache["explain_path"][cache_key] = result
            return result

        except Exception as e:
            print(f"\n[Advertencia] Error al generar explicación: {e}")
            return fallback

    def compare_algorithms(self, greedy_result: dict, beam_result: dict,
                           astar_result: dict, goal_summary: str) -> dict:
        """Usa el LLM para recomendar el mejor algoritmo entre los tres resultados."""
        fallback = {
            "recommended": "a_star",
            "reason": "No se pudo analizar con IA. A* generalmente produce rutas más eficientes.",
            "tradeoff": "Ver métricas numéricas para comparar cobertura y horas."
        }

        # Verificar caché
        cache_key = self._make_cache_key((
            goal_summary,
            tuple(greedy_result["path"]),
            tuple(beam_result["path"]),
            tuple(astar_result["path"])
        ))
        if cache_key in self._cache["compare_algorithms"]:
            return self._cache["compare_algorithms"][cache_key]

        try:
            prompt = (
                "Eres un sistema de IA que ayuda a elegir rutas de aprendizaje. "
                "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                "sin comillas de código, sin explicaciones.\n\n"
                f"El usuario quiere: {goal_summary}\n\n"
                "Se generaron tres rutas con algoritmos distintos:\n\n"
                f"GREEDY:\n"
                f"- Horas totales: {greedy_result['total_hours']}h\n"
                f"- Cobertura: {greedy_result['coverage_pct']}%\n"
                f"- Recursos: {len(greedy_result['path'])}\n"
                f"- Habilidades cubiertas: {', '.join(greedy_result['skills_covered'])}\n"
                f"- Habilidades faltantes: {', '.join(greedy_result['skills_missing']) or 'ninguna'}\n\n"
                f"BEAM SEARCH:\n"
                f"- Horas totales: {beam_result['total_hours']}h\n"
                f"- Cobertura: {beam_result['coverage_pct']}%\n"
                f"- Recursos: {len(beam_result['path'])}\n"
                f"- Habilidades cubiertas: {', '.join(beam_result['skills_covered'])}\n"
                f"- Habilidades faltantes: {', '.join(beam_result['skills_missing']) or 'ninguna'}\n\n"
                f" A* :\n"
                f"- Horas totales: {astar_result['total_hours']}h\n"
                f"- Cobertura: {astar_result['coverage_pct']}%\n"
                f"- Recursos: {len(astar_result['path'])}\n"
                f"- Habilidades cubiertas: {', '.join(astar_result['skills_covered'])}\n"
                f"- Habilidades faltantes: {', '.join(astar_result['skills_missing']) or 'ninguna'}\n\n"
                "Responde ÚNICAMENTE con un JSON válido:\n\n"
                "{\n"
                "  \"recommended\": \"greedy\", \"beam_search\" o \"a_star\",\n"
                "  \"reason\": \"explicación breve de por qué uno es mejor para este caso\",\n"
                "  \"tradeoff\": \"qué sacrifica cada algoritmo en este caso concreto\"\n"
                "}"
            )

            raw = self._call_llm(prompt)
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)

            self._cache["compare_algorithms"][cache_key] = result
            return result

        except json.JSONDecodeError as e:
            print(f"\n[Advertencia] El LLM devolvió una respuesta no válida al comparar algoritmos: {e}")
            print("Continuando con recomendación por defecto...")
            return fallback
        except Exception as e:
            print(f"\n[Advertencia] Error al conectar con el LLM: {e}")
            print("Continuando con recomendación por defecto...")
            return fallback

    def score_resources_for_goal(self, goal_summary: str,
                                  resources: list[dict]) -> dict[str, float]:
        """Evalúa la relevancia de cada recurso para el objetivo del usuario."""
        fallback = {r["id"]: 0.5 for r in resources}

        # Verificar caché
        resource_ids = tuple(sorted([r["id"] for r in resources]))
        cache_key = self._make_cache_key((goal_summary, resource_ids))
        if cache_key in self._cache["score_resources"]:
            return self._cache["score_resources"][cache_key]

        try:
            resources_str = "\n".join(
                f"- id={r['id']} nombre='{r['name']}' dominio={r['domain']} enseña={', '.join(r['teaches'])}"
                for r in resources
            )
            prompt = (
                "Eres un evaluador de recursos educativos. "
                "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                "sin comillas de código, sin explicaciones.\n\n"
                f"El usuario tiene este objetivo de aprendizaje: \"{goal_summary}\"\n\n"
                "Evalúa qué tan relevante es cada recurso para alcanzar ese objetivo.\n"
                "Asigna un puntaje entre 0.0 (irrelevante) y 1.0 (esencial).\n\n"
                f"Recursos a evaluar:\n{resources_str}\n\n"
                "Responde ÚNICAMENTE con un JSON válido con este formato exacto:\n"
                "{\n"
                "  \"scores\": {\n"
                "    \"r01\": 0.9,\n"
                "    \"r02\": 0.4,\n"
                "    ...\n"
                "  }\n"
                "}\n\n"
                "Incluye todos los IDs de la lista. Sé preciso: no todo puede ser 1.0."
            )

            raw = self._call_llm(prompt)
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            scores = parsed.get("scores", {})

            # Rellenar con 0.5 cualquier recurso que el LLM haya omitido
            for r in resources:
                if r["id"] not in scores:
                    scores[r["id"]] = 0.5

            self._cache["score_resources"][cache_key] = scores
            return scores

        except json.JSONDecodeError as e:
            print(f"\n[Advertencia] El LLM devolvió una respuesta no válida al evaluar recursos: {e}")
            print("Continuando con scores neutros...")
            return fallback
        except Exception as e:
            print(f"\n[Advertencia] Error al conectar con el LLM: {e}")
            print("Continuando con scores neutros...")
            return fallback