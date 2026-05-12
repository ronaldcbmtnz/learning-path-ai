from src.llm_client import LLMClient

llm = LLMClient()
result = llm.parse_user_goal('quiero aprender machine learning desde cero, solo se programacion basica y tengo unas 60 horas disponibles')
print(result)