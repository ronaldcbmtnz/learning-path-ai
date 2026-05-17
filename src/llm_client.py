import os
import json
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"
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

    def parse_user_goal(self, user_input: str, available_skills: list) -> dict:
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
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un extractor de información estructurada. "
                            "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                            "sin comillas de código, sin explicaciones."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
    El usuario quiere generar una ruta de aprendizaje personalizada.
    Su objetivo en sus propias palabras es: "{user_input}"

    IMPORTANTE: Solo puedes usar habilidades de esta lista exacta:
    {skills_str}

    Extrae la siguiente información y responde ÚNICAMENTE con un JSON válido:

    {{
    "target_skills": ["habilidades de la lista que mejor representan lo que quiere aprender"],
    "known_skills": ["habilidades de la lista que ya menciona tener, vacío si no menciona ninguna"],
    "max_hours": null o un número si menciona límite de tiempo,
    "goal_summary": "resumen del objetivo en una oración"
    }}

    Reglas estrictas:
    - Usa ÚNICAMENTE habilidades que aparezcan en la lista proporcionada
    - Si el usuario menciona "machine learning", mapéalo a las habilidades relevantes de la lista
    - Si el usuario dice "no sé nada", known_skills debe ser una lista vacía
    - Selecciona todas las habilidades relevantes al objetivo, no solo una
    """
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            # Validar que las claves esperadas existen
            for key in ["target_skills", "known_skills", "max_hours", "goal_summary"]:
                if key not in result:
                    result[key] = fallback[key]
            
            # Guardar en caché
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
        fallback = (
            f"Tu ruta de aprendizaje cubre el {path_result['coverage_pct']}% de tu objetivo "
            f"en {path_result['total_hours']} horas con {len(path_result['path'])} recursos. "
            "Sigue la secuencia propuesta respetando el orden, ya que cada recurso "
            "construye sobre el anterior."
        )
        try:
            resources_detail = []
            for rid in path_result["path"]:
                r = get_resource_fn(rid)
                resources_detail.append(
                    f"- {r['name']} ({r['duration_hours']}h, {r['type']}): "
                    f"enseña {', '.join(r['teaches'])}"
                )
            resources_str = "\n".join(resources_detail)

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un tutor experto en aprendizaje personalizado."
                    },
                    {
                        "role": "user",
                        "content": f"""
    El usuario quiere: {goal_summary}

    El sistema generó esta ruta de aprendizaje usando el algoritmo {path_result['algorithm']}:

    {resources_str}

    Duración total: {path_result['total_hours']} horas
    Cobertura del objetivo: {path_result['coverage_pct']}%

    Escribe una explicación motivadora y clara para el usuario. Debe incluir:
    1. Por qué esta secuencia tiene sentido (menciona las dependencias clave)
    2. Qué va a lograr al final
    3. Un consejo práctico para mantener el ritmo

    Máximo 200 palabras. Habla directamente al usuario (usa "tú").
    """
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n[Advertencia] Error al generar explicación: {e}")
            return fallback
    
    def compare_algorithms(self, greedy_result: dict, beam_result: dict,
                           astar_result: dict, goal_summary: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un sistema de IA que ayuda a elegir rutas de aprendizaje. "
                        "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                        "sin comillas de código, sin explicaciones."
                    )
                },
                {
                    "role": "user",
                    "content": f"""
El usuario quiere: {goal_summary}

Se generaron tres rutas con algoritmos distintos:

GREEDY:
- Horas totales: {greedy_result['total_hours']}h
- Cobertura: {greedy_result['coverage_pct']}%
- Recursos: {len(greedy_result['path'])}
- Habilidades cubiertas: {', '.join(greedy_result['skills_covered'])}
- Habilidades faltantes: {', '.join(greedy_result['skills_missing']) or 'ninguna'}

BEAM SEARCH:
- Horas totales: {beam_result['total_hours']}h
- Cobertura: {beam_result['coverage_pct']}%
- Recursos: {len(beam_result['path'])}
- Habilidades cubiertas: {', '.join(beam_result['skills_covered'])}
- Habilidades faltantes: {', '.join(beam_result['skills_missing']) or 'ninguna'}

A* (ÓPTIMO):
- Horas totales: {astar_result['total_hours']}h
- Cobertura: {astar_result['coverage_pct']}%
- Recursos: {len(astar_result['path'])}
- Habilidades cubiertas: {', '.join(astar_result['skills_covered'])}
- Habilidades faltantes: {', '.join(astar_result['skills_missing']) or 'ninguna'}

Responde ÚNICAMENTE con un JSON válido:

{{
  "recommended": "greedy", "beam_search" o "a_star",
  "reason": "explicación breve de por qué uno es mejor para este caso",
  "tradeoff": "qué sacrifica cada algoritmo en este caso concreto"
}}
"""
                }
            ]
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    
    
    def score_resources_for_goal(self, goal_summary: str, resources: list) -> dict:
        # Fallback: score neutro 0.5 para todos los recursos
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
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un evaluador de recursos educativos. "
                            "Respondes ÚNICAMENTE con JSON válido, sin texto adicional, "
                            "sin comillas de código, sin explicaciones."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"""
    El usuario tiene este objetivo de aprendizaje: "{goal_summary}"

    Evalúa qué tan relevante es cada recurso para alcanzar ese objetivo.
    Asigna un puntaje entre 0.0 (irrelevante) y 1.0 (esencial).

    Recursos a evaluar:
    {resources_str}

    Responde ÚNICAMENTE con un JSON válido con este formato exacto:
    {{
    "scores": {{
        "r01": 0.9,
        "r02": 0.4,
        ...
    }}
    }}

    Incluye todos los IDs de la lista. Sé preciso: no todo puede ser 1.0.
    """
                    }
                ]
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            scores = parsed.get("scores", {})
            # Rellenar con 0.5 cualquier recurso que el LLM haya omitido
            for r in resources:
                if r["id"] not in scores:
                    scores[r["id"]] = 0.5
            
            # Guardar en caché
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