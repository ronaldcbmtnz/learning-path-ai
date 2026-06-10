"""
Casos de prueba para evaluar ambos algoritmos bajo distintos perfiles de usuario.
Cada caso define: habilidades objetivo, habilidades previas, horas disponibles,
y una descripción del perfil para el informe.
"""

TEST_CASES = [
    {
        "id": "TC01",
        "profile": "Principiante absoluto con tiempo limitado",
        "target_skills": {"python_basico", "estadistica"},
        "known_skills": set(),
        "max_hours": 20,
    },
    {
        "id": "TC02",
        "profile": "Principiante que quiere entrar a ML con tiempo amplio",
        "target_skills": {"ml_supervisado", "evaluacion_modelos", "sklearn"},
        "known_skills": set(),
        "max_hours": 80,
    },
    {
        "id": "TC03",
        "profile": "Programador que quiere especializarse en datos",
        "target_skills": {"pandas", "analisis_exploratorio", "sql", "dashboards"},
        "known_skills": {"python_basico", "funciones"},
        "max_hours": 30,
    },
    {
        "id": "TC04",
        "profile": "Estudiante con base matemática que quiere ML avanzado",
        "target_skills": {"redes_neuronales", "backpropagation", "pytorch"},
        "known_skills": {"algebra_lineal", "calculo", "probabilidad"},
        "max_hours": 60,
    },
    {
        "id": "TC05",
        "profile": "Desarrollador que quiere desplegar modelos en producción",
        "target_skills": {"mlops", "docker", "api_modelos"},
        "known_skills": {"python_basico", "funciones", "ml_supervisado"},
        "max_hours": 20,
    },
    {
        "id": "TC06",
        "profile": "Principiante con tiempo muy restrictivo",
        "target_skills": {"ml_supervisado", "redes_neuronales"},
        "known_skills": set(),
        "max_hours": 15,
    },
    {
        "id": "TC07",
        "profile": "Científico de datos que quiere visión artificial",
        "target_skills": {"computer_vision", "cnns", "deteccion_objetos"},
        "known_skills": {"python_basico", "ml_supervisado", "estadistica"},
        "max_hours": 70,
    },
    {
        "id": "TC08",
        "profile": "Analista que quiere automatizar reportes",
        "target_skills": {"pandas", "streamlit", "visualizacion_web"},
        "known_skills": {"python_basico", "estadistica"},
        "max_hours": 15,
    },
    {
        "id": "TC09",
        "profile": "Desarrollador que quiere entrar a NLP",
        "target_skills": {"nlp", "transformers", "embeddings"},
        "known_skills": {"python_basico", "ml_supervisado"},
        "max_hours": 80,
    },
    {
        "id": "TC10",
        "profile": "Principiante que quiere dominar programación desde cero",
        "target_skills": {"python_basico", "poo", "estructuras_datos", "git"},
        "known_skills": set(),
        "max_hours": 40,
    },
    {
        "id": "TC11",
        "profile": "ML engineer que quiere ir a cloud",
        "target_skills": {"cloud_basico", "storage", "compute"},
        "known_skills": {"python_basico", "apis_rest", "ml_supervisado"},
        "max_hours": 10,
    },
    {
        "id": "TC12",
        "profile": "Estudiante con todo el tiempo del mundo",
        "target_skills": {"ml_supervisado", "redes_neuronales", "nlp",
                          "computer_vision", "mlops"},
        "known_skills": set(),
        "max_hours": 200,
    },
    {
        "id": "TC13",
        "profile": "Bachiller que quiere repasar conceptos básicos antes de universidad",
        "target_skills": {"python_basico", "logica_programacion", "funciones"},
        "known_skills": {"funciones"},
        "max_hours": 50,
    },
    {
        "id": "TC14",
        "profile": "Ingeniero con ML que quiere especializarse en análisis de datos avanzado",
        "target_skills": {"json", "numpy", "consultas", "poo", "streamlit", "storage"},
        "known_skills": {"ml_supervisado"},
        "max_hours": 56,
    },
    # --- Fase 4: casos in-domain sobre las cadenas nuevas del catálogo ampliado ---
    {
        "id": "TC15",
        "profile": "Aspirante a desarrollador web full-stack desde cero",
        "target_skills": {"react", "fastapi", "testing"},
        "known_skills": set(),
        "max_hours": 80,  # min 69h -> factible; cadena profunda (js<-logica, react<-js+html+css)
    },
    {
        "id": "TC16",
        "profile": "Desarrollador que quiere dar el salto a DevOps/Cloud",
        "target_skills": {"kubernetes", "ci_cd", "terraform"},
        "known_skills": {"git"},
        "max_hours": 90,  # min 76h -> factible; k8s<-contenedores<-linux, ci_cd<-testing
    },
    {
        "id": "TC17",
        "profile": "ML engineer que quiere especializarse en LLMs y sistemas RAG",
        "target_skills": {"llms", "rag", "fine_tuning"},
        "known_skills": {"python_basico", "ml_supervisado"},
        "max_hours": 110,  # min 102h -> factible; cadena MUY profunda (rag<-llms<-transformer<-redes)
    },
    {
        "id": "TC18",
        "profile": "Analista que quiere convertirse en ingeniero de datos",
        "target_skills": {"spark", "data_warehouse", "airflow"},
        "known_skills": {"python_basico", "sql"},
        "max_hours": 45,  # min 39h -> factible; todo cuelga de etl
    },
    {
        "id": "TC19",
        "profile": "Data scientist que quiere dominar ML clásico con presupuesto justo",
        "target_skills": {"random_forest", "svm", "clustering", "series_temporales"},
        "known_skills": {"python_basico", "estadistica"},
        "max_hours": 90,  # min 96h -> INFACTIBLE por presupuesto (mejor parcial)
    },
]