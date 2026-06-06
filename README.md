# Learning Path AI

AI-powered personalized learning path generator that combines graph optimization algorithms with large language models to create tailored educational routes.

## What it does

The user describes their learning goal in plain language. The system then:

1. Uses an LLM to extract target skills and constraints from the natural language input
2. Scores resources based on goal relevance using the LLM
3. Builds a dependency graph of learning resources with difficulty levels
4. Runs three optimization algorithms (Greedy, Beam Search, A*) to find the best learning sequence
5. Uses the LLM to compare all three routes and recommend the optimal one
6. Generates a motivational explanation of the recommended path

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
- OpenRouter API (LLaMA 3.3 70B Instruct) — natural language understanding + resource scoring
- Custom graph engine — dependency resolution, cycle detection, topological sorting
- Three optimization algorithms — Greedy, Beam Search, and A* (heuristic-based)
- Difficulty-aware path optimization — penalizes abrupt difficulty jumps
- LLM response caching — reduces API calls and costs

## Project structure

learning-path-ai/  
├── data/  
│   ├── resources.json            # Dataset of 20 learning resources with skills, dependencies, and difficulty
│   └── evaluation_results.json   # Benchmark results from evaluator  
├── src/  
│   ├── graph.py                  # ResourceGraph: cycle detection, dependency resolution, topological sort    
│   ├── optimizer.py              # PathOptimizer: Greedy, Beam Search, and A* algorithms
│   ├── llm_client.py             # LLMClient: OpenRouter API integration with caching
│   └── main.py                   # Main pipeline connecting all components  
├── tests/  
│   ├── evaluator.py              # Comparative benchmarking of all algorithms
│   ├── test_cases.py             # 14 test profiles for evaluation
│   
└── CAMBIOS_IMPLEMENTADOS.md      # Detailed changelog of improvements  

## How to run

```bash
# Install dependencies
pip install openai python-dotenv

# Add your Groq API key to .env
echo "OPENROUTER_API_KEY=your-key-here" > .env

# Run
python -m src.main
```

## Algorithms

Three algorithms are implemented and compared:

**Greedy** — At each step selects the resource that maximizes future target coverage 
using forward simulation. Fast (avg 0.5ms) with good coverage. Considers resource difficulty 
and penalizes abrupt difficulty jumps to ensure smooth learning progression.
Achieves 100% coverage in 6/14 test cases.

**Beam Search** — Maintains the K best partial paths simultaneously (beam width 3-6), 
exploring more of the solution space before committing. Similar coverage to Greedy 
but uses fewer hours on average. Also respects difficulty levels. 
Achieves 100% coverage in 6/14 test cases.

**A* (heuristic)** — Heuristic-based search using a non-admissible heuristics 
on missing skills and resource difficulty. Explores paths more intelligently than Greedy/Beam.
Outperforms both in coverage while maintaining efficiency. 
Trade-off: slightly slower on large search spaces but still practical (avg 0.7ms).
Achieves 100% coverage in 8/14 test cases.

### Benchmark Results (14 test cases)

| Algorithm   | Avg Coverage | Avg Hours | Perfect (100%) | Avg Time |
|-------------|-------------|-----------|----------------|----------|
| Greedy      | 57.7%       | 45.4h     | 6/14           | 0.5ms    |
| Beam Search | 57.7%       | 28.6h     | 6/14           | 2.0ms    |
| A* (heuristic)| **66.1%** | **25.3h** | **8/14**       | **0.7ms**|

## Optimizations

### Cycle Detection
- Validates learning resource graph on load to detect circular dependencies
- Prevents infinite loops in prerequisite resolution
- Raises clear error messages with cycle details

### LLM Caching
- Caches API responses using SHA256 hashing on inputs
- Reduces API calls by ~70% on repeated evaluations
- Transparent fallback to default scores if API unavailable

### Difficulty-Aware Paths
- Resources have difficulty levels (1=basic, 2=intermediate, 3=advanced)
- Algorithms penalize jumping more than 1 difficulty level at once
- Produces smoother, more pedagogically sound learning paths
- Penalty formula: `max(0, difficulty - max_so_far - 1) * 5`

### Optimized A* Search
- Filters candidates to only resources that help achieve targets
- 2-3x performance improvement over naive A* implementation
- Explores more intelligently than Greedy/Beam while keeping runtime practical.

## Evaluation & Testing

### Run Evaluator
```bash
# Run comparative benchmark on all 14 test cases
python -m tests.evaluator

# Output: CSV-style results with algorithm, coverage, hours, and timing
# Saves results to: data/evaluation_results.json
```

### Test Coverage
- **14 diverse profiles**: From absolute beginners to ML engineers
- **Realistic scenarios**: Time constraints, difficulty progressions, various domains
- **Reproducible**: Fixed test cases with known expected behaviors

## Recent Improvements (May 2026)

### Performance
- **A* Optimization**: Reduced average time from 13ms to 0.7ms (18x faster)
- **Candidate Filtering**: Smart resource selection based on goal relevance
- **API Efficiency**: 70% reduction in LLM calls via intelligent caching

### Robustness
- **Cycle Detection**: Validates graph integrity on initialization
- **Error Handling**: Graceful fallbacks when LLM unavailable
- **Per-Test-Case Scoring**: Each evaluation uses fresh LLM relevance scores

### Quality
- **Difficulty-Aware**: Smoother learning progressions (no jarring jumps)
- **Better Coverage**: A* achieves 66.1% avg coverage (vs 57.7% for Greedy/Beam)
- **More Efficient**: A* uses 25.3h on average (vs 45.4h for Greedy)

## License

MIT License - See LICENSE file for details
