import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def parse_user_goal(self, user_input: str, available_skills: list) -> dict:
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
        return json.loads(raw)

    def explain_path(self, path_result: dict, goal_summary: str,
                     get_resource_fn) -> str:
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