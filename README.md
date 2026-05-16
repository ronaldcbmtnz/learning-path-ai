# Learning Path AI

AI-powered personalized learning path generator that combines graph optimization algorithms with large language models to create tailored educational routes.

## What it does

The user describes their learning goal in plain language. The system then:

1. Uses an LLM to extract target skills and constraints from the natural language input
2. Builds a dependency graph of learning resources
3. Runs two optimization algorithms (Greedy and Beam Search) to find the best learning sequence
4. Uses the LLM to compare both routes and recommend the best one
5. Generates a motivational explanation of the recommended path

## Example

Input: "I want to learn advanced machine learning, I know nothing and have 8 hours a day for 7 days"  
Output:  
Algorithm : GREEDY  
Hours     : 56h  
Coverage  : 58.3%  
Path:  
1. Python for beginners (10h)
2. Mathematics for Machine Learning (15h)
3. Applied Statistics with Python (8h)
4. Introduction to Machine Learning (20h)

## Tech stack

- Python 3.13
- Groq API (LLaMA 3.3 70B) — natural language understanding
- Custom graph engine — dependency resolution and topological sorting
- Greedy and Beam Search algorithms — path optimization

## Project structure

learning-path-ai/  
├── data/  
│   └── resources.json       # Dataset of 20 learning resources with skills and   dependencies  
├── src/  
│   ├── graph.py             # ResourceGraph: dependency detection and topological sort    
│   ├── optimizer.py         # PathOptimizer: Greedy and Beam Search algorithms  
│   ├── llm_client.py        # LLMClient: Groq API integration  
│   └── main.py              # Main pipeline connecting all components  
└── tests/  

## How to run

```bash
# Install dependencies
pip install groq python-dotenv

# Add your Groq API key to .env
echo "GROQ_API_KEY=your-key-here" > .env

# Run
python -m src.main
```

## Algorithms

Three algorithms are implemented and compared:

**Greedy** — at each step selects the resource that maximizes future target coverage 
using a forward simulation. Fast (avg 0.8ms) but commits to early decisions 
without backtracking. Achieves 100% coverage in 5/12 test cases.

**Beam Search** — maintains the K best partial paths simultaneously (beam width 3-6), 
exploring more of the solution space before committing. Similar coverage to greedy 
but uses fewer hours on average. Achieves 100% coverage in 5/12 test cases.

**A\* (recommended)** — guarantees the optimal path using an admissible heuristic 
(number of missing target skills). Finds maximum coverage with minimum hours. 
Achieves 100% coverage in 7/12 test cases, outperforming both heuristic approaches. 
Trade-off: slower on large search spaces (avg 13ms).

| Algorithm   | Avg Coverage | Avg Hours | Perfect (100%) |
|-------------|-------------|-----------|----------------|
| Greedy      | 52.1%       | 40.2h     | 5/12           |
| Beam Search | 52.1%       | 27.9h     | 5/12           |
| A*          | 72.1%       | 28.0h     | 7/12           |
