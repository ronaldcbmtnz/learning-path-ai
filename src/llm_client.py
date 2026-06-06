import os
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model_name = os.getenv(
            "OPENROUTER_MODEL",
            "meta-llama/llama-3.3-70b-instruct:free"
        )
        # Caché para evitar múltiples llamadas a la API con los mismos inputs
        self._cache = {
            "parse_user_goal": {},
            "score_resources": {},
            "compare_algorithms": {},
            "explain_path": {}
        }

    def _make_cache_key(self, data) -> str:
        """Genera una clave de caché usando hash SHA256 del input."""
        if isinstance(data, (list, dict, set)):
            data = json.dumps(data, sort_keys=True, default=str)
        elif not isinstance(data, str):
            data = str(data)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _call_llm(self, prompt: str) -> str:
        """Llamada base al LLM. Lanza excepción si falla."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def parse_user_goal(self, user_input: str, available_skills: list) -> dict:
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
                     get_resource_fn) -> str:
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
                f"A* (heurístico):\n"
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

    def score_resources_for_goal(self, goal_summary: str, resources: list) -> dict:
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