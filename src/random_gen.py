"""
Generadores de variables aleatorias por transformada inversa (modulo Simulacion,
Cap. 2). Puros y SEMILLABLES: misma semilla => misma secuencia (reproducibilidad,
HANDOFF_SIMULACION.md §3.5).

NO importa nada del modulo de IA: es una pieza matematica autonoma. La usa
src/simulation.py para modelar el ruido del LLM.

Metodo de la transformada inversa (Cap. 2.1): si U ~ Uniforme(0,1) y F es la CDF
de la variable deseada, entonces X = F^{-1}(U) tiene esa distribucion. Aqui se
implementa en forma cerrada para la uniforme, la triangular simetrica (el modelo
de ruido) y, como demostracion extra de 2.3, la exponencial.
"""
import math
import random


class RandomGen:
    """Fuente de aleatoriedad semillada. Envuelve random.Random para fijar la
    secuencia base de U(0,1) y construir sobre ella las demas distribuciones por
    transformada inversa."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self.seed = seed

    def reset(self, seed: int | None = None) -> None:
        """Reinicia la secuencia (a la semilla dada o a la original)."""
        if seed is not None:
            self.seed = seed
        self._rng.seed(self.seed)

    # ------------------------------------------------------------------
    # U(0,1): la fuente primitiva de todas las transformadas inversas.
    # ------------------------------------------------------------------
    def u01(self) -> float:
        return self._rng.random()

    # ------------------------------------------------------------------
    # Uniforme(a, b)  (Cap. 2.2):  X = a + (b - a) * U
    # ------------------------------------------------------------------
    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.u01()

    # ------------------------------------------------------------------
    # Triangular simetrica en [-s, +s] con moda 0  (Cap. 2.1, transf. inversa).
    #
    # Es el MODELO DE RUIDO del LLM: error pequeno y centrado, mas probable cerca
    # de 0 que en los extremos. CDF por tramos -> inversa en forma cerrada:
    #   U < 0.5 :  x = -s + s * sqrt(2U)
    #   U >= 0.5:  x =  s - s * sqrt(2(1-U))
    # Comprobacion: U=0 -> -s ; U=0.5 -> 0 ; U=1 -> +s ; E[X] = 0 por simetria.
    # ------------------------------------------------------------------
    def triangular_sym(self, s: float) -> float:
        if s <= 0:
            return 0.0
        u = self.u01()
        if u < 0.5:
            return -s + s * math.sqrt(2.0 * u)
        return s - s * math.sqrt(2.0 * (1.0 - u))

    # ------------------------------------------------------------------
    # Exponencial(lambda)  (Cap. 2.3, demostracion extra):  X = -ln(1 - U) / lambda
    # No se usa para el ruido; se incluye como ejemplo clasico de transf. inversa.
    # ------------------------------------------------------------------
    def exponential(self, lam: float) -> float:
        if lam <= 0:
            raise ValueError("lambda debe ser > 0")
        return -math.log(1.0 - self.u01()) / lam
