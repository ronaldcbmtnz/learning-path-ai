from src.graph import ResourceGraph
from src.optimizer import PathOptimizer

g = ResourceGraph()
opt = PathOptimizer(g)

target = {"ml_supervisado", "redes_neuronales", "estadistica"}
result = opt.compare(target, max_hours=80)

for algo, r in result.items():
    print(f"\n=== {algo.upper()} ===")
    print(f"Horas totales : {r['total_hours']}h")
    print(f"Cobertura     : {r['coverage_pct']}%")
    print("Ruta:")
    for rid in r["path"]:
        rec = g.get_resource(rid)
        print(f"  {rec['name']} ({rec['duration_hours']}h)")
